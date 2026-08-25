#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

namespace {

using Polygon = std::vector<cv::Point2f>;

struct CachedFrameContext {
  cv::Mat gt_mask;
  cv::Mat pred_mask;
  cv::Mat intersection_mask;
  std::int64_t gt_area = 0;
  float shift_x = 0.0F;
  float shift_y = 0.0F;
  float scale_factor = 1.0F;
};

struct CachedMetricsTotals {
  double frame_loss_total = 0.0;
  double recall_deficit_total = 0.0;
  int frames_covered = 0;
};

struct RasterRun {
  int y = 0;
  int start_x = 0;
  int end_x = 0;
};

struct ExactReferenceVariant {
  std::vector<RasterRun> runs;
  std::int64_t area = 0;
};

struct ExactFrameReference {
  bool has_polygon = false;
  float min_x = std::numeric_limits<float>::infinity();
  float min_y = std::numeric_limits<float>::infinity();
  float max_x = -std::numeric_limits<float>::infinity();
  float max_y = -std::numeric_limits<float>::infinity();
  std::array<ExactReferenceVariant, 4> variants;
};

struct ExactRasterScratch {
  std::vector<std::uint8_t> prediction;
};

struct ExactMetricCounts {
  std::int64_t gt_area = 0;
  std::int64_t pred_area = 0;
  std::int64_t intersection = 0;
  std::int64_t union_area = 0;
  double recall = 1.0;
  double precision = 1.0;
  double iou = 1.0;
};

bool has_strict_self_intersection_impl(
    const double* points,
    const std::size_t point_count) {
  if (point_count < 4U) {
    return false;
  }
  struct EdgeBounds {
    std::size_t index = 0U;
    double minimum_x = 0.0;
    double minimum_y = 0.0;
    double maximum_x = 0.0;
    double maximum_y = 0.0;
  };
  std::vector<EdgeBounds> edges(point_count);
  for (std::size_t index = 0; index < point_count; ++index) {
    const std::size_t following = (index + 1U) % point_count;
    const double a_x = points[index * 2U];
    const double a_y = points[index * 2U + 1U];
    const double b_x = points[following * 2U];
    const double b_y = points[following * 2U + 1U];
    if (!std::isfinite(a_x) || !std::isfinite(a_y) ||
        !std::isfinite(b_x) || !std::isfinite(b_y)) {
      throw std::invalid_argument("polygon coordinates must be finite");
    }
    edges[index] = {
        index,
        std::min(a_x, b_x),
        std::min(a_y, b_y),
        std::max(a_x, b_x),
        std::max(a_y, b_y)};
  }
  std::stable_sort(
      edges.begin(),
      edges.end(),
      [](const EdgeBounds& left, const EdgeBounds& right) {
        return left.minimum_x < right.minimum_x;
      });
  std::vector<std::size_t> active;
  active.reserve(point_count);
  for (const auto& first_bounds : edges) {
    active.erase(
        std::remove_if(
            active.begin(),
            active.end(),
            [&](const std::size_t second) {
              return edges[second].maximum_x < first_bounds.minimum_x;
            }),
        active.end());
    const std::size_t first = first_bounds.index;
    const std::size_t first_following = (first + 1U) % point_count;
    const double a_x = points[first * 2U];
    const double a_y = points[first * 2U + 1U];
    const double b_x = points[first_following * 2U];
    const double b_y = points[first_following * 2U + 1U];
    for (const std::size_t active_position : active) {
      const EdgeBounds& second_bounds = edges[active_position];
      const std::size_t second = second_bounds.index;
      if ((second + 1U) % point_count == first || first_following == second) {
        continue;
      }
      if (std::max(first_bounds.minimum_y, second_bounds.minimum_y) >
          std::min(first_bounds.maximum_y, second_bounds.maximum_y)) {
        continue;
      }
      const std::size_t second_following = (second + 1U) % point_count;
      const double c_x = points[second * 2U];
      const double c_y = points[second * 2U + 1U];
      const double d_x = points[second_following * 2U];
      const double d_y = points[second_following * 2U + 1U];
      const double ab_x = b_x - a_x;
      const double ab_y = b_y - a_y;
      const double cd_x = d_x - c_x;
      const double cd_y = d_y - c_y;
      const double ab_c = ab_x * (c_y - a_y) - ab_y * (c_x - a_x);
      const double ab_d = ab_x * (d_y - a_y) - ab_y * (d_x - a_x);
      const double cd_a = cd_x * (a_y - c_y) - cd_y * (a_x - c_x);
      const double cd_b = cd_x * (b_y - c_y) - cd_y * (b_x - c_x);
      if (ab_c * ab_d < -1e-8 && cd_a * cd_b < -1e-8) {
        return true;
      }
    }
    // Store the position in the x-sorted edge vector.  This lets expiration
    // reuse the already-computed bounds without changing the stable sweep
    // order used by the Python reference implementation.
    active.push_back(
        static_cast<std::size_t>(&first_bounds - edges.data()));
  }
  return false;
}

py::array_t<std::uint8_t> strict_self_intersection_batch(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& values,
    const int requested_threads) {
  if (values.ndim() != 3 || values.shape(2) != 2) {
    throw py::value_error("values must have shape (N, points, 2)");
  }
  const py::ssize_t case_count = values.shape(0);
  const py::ssize_t point_count = values.shape(1);
  py::array_t<std::uint8_t> output(case_count);
  const double* input = values.data();
  std::uint8_t* result = output.mutable_data();
  const std::size_t values_per_case =
      static_cast<std::size_t>(point_count) * 2U;
  const int thread_count = std::max(1, requested_threads);
  {
    py::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 64) num_threads(thread_count)
#endif
    for (py::ssize_t index = 0; index < case_count; ++index) {
      result[index] = has_strict_self_intersection_impl(
          input + static_cast<std::size_t>(index) * values_per_case,
          static_cast<std::size_t>(point_count))
          ? 1U
          : 0U;
    }
  }
  return output;
}

bool polygon_is_simple_impl(
    const double* points,
    const std::size_t point_count) {
  constexpr double epsilon = 1e-8;
  if (point_count < 3U) {
    return false;
  }
  double signed_twice_area = 0.0;
  for (std::size_t index = 0; index < point_count; ++index) {
    const std::size_t following = (index + 1U) % point_count;
    const double x = points[index * 2U];
    const double y = points[index * 2U + 1U];
    const double next_x = points[following * 2U];
    const double next_y = points[following * 2U + 1U];
    if (!std::isfinite(x) || !std::isfinite(y) ||
        !std::isfinite(next_x) || !std::isfinite(next_y)) {
      return false;
    }
    if (std::hypot(next_x - x, next_y - y) <= epsilon) {
      return false;
    }
    signed_twice_area += x * next_y - next_x * y;
  }
  if (std::abs(signed_twice_area) <= epsilon) {
    return false;
  }

  const auto orientation = [](const double* a, const double* b, const double* c) {
    return (b[0] - a[0]) * (c[1] - a[1]) -
           (b[1] - a[1]) * (c[0] - a[0]);
  };
  const auto on_segment = [epsilon](
                              const double* a,
                              const double* b,
                              const double* point) {
    return std::min(a[0], b[0]) - epsilon <= point[0] &&
           point[0] <= std::max(a[0], b[0]) + epsilon &&
           std::min(a[1], b[1]) - epsilon <= point[1] &&
           point[1] <= std::max(a[1], b[1]) + epsilon;
  };
  for (std::size_t first_edge = 0; first_edge < point_count; ++first_edge) {
    const std::size_t first_following = (first_edge + 1U) % point_count;
    const double* a = points + first_edge * 2U;
    const double* b = points + first_following * 2U;
    for (std::size_t second_edge = first_edge + 1U;
         second_edge < point_count;
         ++second_edge) {
      if (second_edge == first_following ||
          (second_edge + 1U) % point_count == first_edge) {
        continue;
      }
      const std::size_t second_following = (second_edge + 1U) % point_count;
      const double* c = points + second_edge * 2U;
      const double* d = points + second_following * 2U;
      const double first = orientation(a, b, c);
      const double second = orientation(a, b, d);
      const double third = orientation(c, d, a);
      const double fourth = orientation(c, d, b);
      if ((first * second < -epsilon && third * fourth < -epsilon) ||
          (std::abs(first) <= epsilon && on_segment(a, b, c)) ||
          (std::abs(second) <= epsilon && on_segment(a, b, d)) ||
          (std::abs(third) <= epsilon && on_segment(c, d, a)) ||
          (std::abs(fourth) <= epsilon && on_segment(c, d, b))) {
        return false;
      }
    }
  }
  return true;
}

py::array_t<std::uint8_t> polygon_is_simple_batch(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& values,
    const int requested_threads) {
  if (values.ndim() != 3 || values.shape(2) != 2) {
    throw py::value_error("values must have shape (N, points, 2)");
  }
  const py::ssize_t case_count = values.shape(0);
  const py::ssize_t point_count = values.shape(1);
  py::array_t<std::uint8_t> output(case_count);
  const double* input = values.data();
  std::uint8_t* result = output.mutable_data();
  const std::size_t values_per_case =
      static_cast<std::size_t>(point_count) * 2U;
  const int thread_count = std::max(1, requested_threads);
  {
    py::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(thread_count)
#endif
    for (py::ssize_t index = 0; index < case_count; ++index) {
      result[index] = polygon_is_simple_impl(
                          input + static_cast<std::size_t>(index) * values_per_case,
                          static_cast<std::size_t>(point_count))
                          ? 1U
                          : 0U;
    }
  }
  return output;
}

py::array_t<std::int32_t> shared_area_best_cycle(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& costs,
    int target) {
  if (costs.ndim() != 2 || costs.shape(1) != costs.shape(0) + 1) {
    throw py::value_error("costs must have shape (samples, samples + 1)");
  }
  const int samples = static_cast<int>(costs.shape(0));
  target = std::max(3, std::min(target, samples));
  const double* values = costs.data();
  const int stride = samples + 1;
  const double infinity = std::numeric_limits<double>::infinity();
  double best_cost = infinity;
  std::vector<std::int32_t> best_indices;
  std::vector<double> previous(static_cast<std::size_t>(samples + 1));
  std::vector<double> current(static_cast<std::size_t>(samples + 1));
  std::vector<std::int32_t> parents(
      static_cast<std::size_t>(target + 1) *
          static_cast<std::size_t>(samples + 1),
      -1);
  for (int anchor = 0; anchor < samples; ++anchor) {
    std::fill(previous.begin(), previous.end(), infinity);
    previous[0] = 0.0;
    std::fill(parents.begin(), parents.end(), -1);
    for (int edge_count = 1; edge_count <= target; ++edge_count) {
      std::fill(current.begin(), current.end(), infinity);
      const int lower = edge_count;
      const int upper = samples - (target - edge_count);
      for (int end = lower; end <= upper; ++end) {
        double selected_value = infinity;
        int selected_start = -1;
        for (int start = edge_count - 1; start < end; ++start) {
          if (!std::isfinite(previous[static_cast<std::size_t>(start)])) {
            continue;
          }
          const int distance = end - start;
          const int row = (anchor + start) % samples;
          const double candidate =
              previous[static_cast<std::size_t>(start)] +
              values[static_cast<std::size_t>(row) * stride + distance];
          if (candidate < selected_value) {
            selected_value = candidate;
            selected_start = start;
          }
        }
        if (selected_start >= 0) {
          current[static_cast<std::size_t>(end)] = selected_value;
          parents[static_cast<std::size_t>(edge_count) * (samples + 1) + end] =
              selected_start;
        }
      }
      previous.swap(current);
    }
    const double candidate_cost = previous[static_cast<std::size_t>(samples)];
    if (candidate_cost >= best_cost) {
      continue;
    }
    std::vector<std::int32_t> positions(static_cast<std::size_t>(target));
    int end = samples;
    for (int edge_count = target; edge_count > 0; --edge_count) {
      const int start = parents[
          static_cast<std::size_t>(edge_count) * (samples + 1) + end];
      if (start < 0) {
        throw std::runtime_error("broken shared-area DP parent chain");
      }
      positions[static_cast<std::size_t>(edge_count - 1)] = start;
      end = start;
    }
    best_cost = candidate_cost;
    best_indices.resize(static_cast<std::size_t>(target));
    for (int index = 0; index < target; ++index) {
      best_indices[static_cast<std::size_t>(index)] =
          static_cast<std::int32_t>(
              (anchor + positions[static_cast<std::size_t>(index)]) % samples);
    }
  }
  if (best_indices.empty()) {
    throw std::runtime_error("shared-area DP has no solution");
  }
  py::array_t<std::int32_t> output(target);
  std::copy(best_indices.begin(), best_indices.end(), output.mutable_data());
  return output;
}

std::vector<Polygon> parse_polygons(const py::iterable& values) {
  std::vector<Polygon> polygons;
  for (const py::handle value : values) {
    const auto array = py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(value);
    if (!array) {
      throw py::type_error("each polygon must be convertible to a float32 NumPy array");
    }
    if (array.ndim() != 2 || array.shape(1) != 2) {
      throw py::value_error("each polygon must have shape (N, 2)");
    }
    const auto view = array.unchecked<2>();
    Polygon polygon;
    polygon.reserve(static_cast<std::size_t>(array.shape(0)));
    for (py::ssize_t i = 0; i < array.shape(0); ++i) {
      polygon.emplace_back(view(i, 0), view(i, 1));
    }
    polygons.push_back(std::move(polygon));
  }
  return polygons;
}

void rasterize_into(
    const std::vector<Polygon>& polygons,
    cv::Mat& mask,
    const float shift_x,
    const float shift_y) {
  mask.setTo(cv::Scalar(0));
  for (const auto& polygon : polygons) {
    if (polygon.size() < 3) {
      continue;
    }
    std::vector<cv::Point> rounded;
    rounded.reserve(polygon.size());
    for (const auto& point : polygon) {
      // NumPy's float32 subtraction followed by np.round uses round-to-nearest,
      // ties-to-even. nearbyint follows the same default IEEE-754 rounding mode.
      const float shifted_x = point.x - shift_x;
      const float shifted_y = point.y - shift_y;
      rounded.emplace_back(
          static_cast<int>(std::nearbyint(shifted_x)),
          static_cast<int>(std::nearbyint(shifted_y)));
    }
    // Production paints components independently. A single call containing
    // all contours would apply OpenCV's even/odd rule and create false holes.
    const std::vector<std::vector<cv::Point>> one_polygon{std::move(rounded)};
    cv::fillPoly(mask, one_polygon, cv::Scalar(1));
  }
}

cv::Mat rasterize(
    const std::vector<Polygon>& polygons,
    const int height,
    const int width,
    const float shift_x,
    const float shift_y) {
  cv::Mat mask = cv::Mat::zeros(height, width, CV_8UC1);
  rasterize_into(polygons, mask, shift_x, shift_y);
  return mask;
}

int positive_modulo_two(const int value) {
  const int remainder = value % 2;
  return remainder < 0 ? remainder + 2 : remainder;
}

int origin_with_parity(const int value, const int parity) {
  return positive_modulo_two(value) == parity ? value : value - 1;
}

ExactFrameReference build_exact_frame_reference(
    const std::vector<Polygon>& polygons) {
  ExactFrameReference output;
  for (const auto& polygon : polygons) {
    if (polygon.size() < 3) {
      continue;
    }
    output.has_polygon = true;
    for (const auto& point : polygon) {
      output.min_x = std::min(output.min_x, point.x);
      output.min_y = std::min(output.min_y, point.y);
      output.max_x = std::max(output.max_x, point.x);
      output.max_y = std::max(output.max_y, point.y);
    }
  }
  if (!output.has_polygon) {
    return output;
  }
  const int floor_x = static_cast<int>(std::floor(output.min_x));
  const int floor_y = static_cast<int>(std::floor(output.min_y));
  const int maximum_x = static_cast<int>(std::ceil(output.max_x));
  const int maximum_y = static_cast<int>(std::ceil(output.max_y));
  for (int parity_y = 0; parity_y < 2; ++parity_y) {
    for (int parity_x = 0; parity_x < 2; ++parity_x) {
      const int variant_index = parity_x + 2 * parity_y;
      const int origin_x = origin_with_parity(floor_x, parity_x);
      const int origin_y = origin_with_parity(floor_y, parity_y);
      const int width = maximum_x - origin_x + 1;
      const int height = maximum_y - origin_y + 1;
      cv::Mat mask = rasterize(
          polygons,
          height,
          width,
          static_cast<float>(origin_x),
          static_cast<float>(origin_y));
      ExactReferenceVariant& variant =
          output.variants[static_cast<std::size_t>(variant_index)];
      for (int y = 0; y < height; ++y) {
        const std::uint8_t* row = mask.ptr<std::uint8_t>(y);
        int x = 0;
        while (x < width) {
          while (x < width && row[x] == 0) {
            ++x;
          }
          if (x >= width) {
            break;
          }
          const int start = x;
          while (x < width && row[x] != 0) {
            ++x;
          }
          variant.runs.push_back(
              {y + origin_y, start + origin_x, x + origin_x});
          variant.area += static_cast<std::int64_t>(x - start);
        }
      }
    }
  }
  return output;
}

ExactMetricCounts exact_metric_counts_from_reference(
    const ExactFrameReference& reference,
    const std::vector<Polygon>& predicted,
    ExactRasterScratch* scratch) {
  bool predicted_has_polygon = false;
  float predicted_min_x = std::numeric_limits<float>::infinity();
  float predicted_min_y = std::numeric_limits<float>::infinity();
  float predicted_max_x = -std::numeric_limits<float>::infinity();
  float predicted_max_y = -std::numeric_limits<float>::infinity();
  for (const auto& polygon : predicted) {
    if (polygon.size() < 3) {
      continue;
    }
    predicted_has_polygon = true;
    for (const auto& point : polygon) {
      if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
        throw std::invalid_argument("polygon coordinates must be finite");
      }
      predicted_min_x = std::min(predicted_min_x, point.x);
      predicted_min_y = std::min(predicted_min_y, point.y);
      predicted_max_x = std::max(predicted_max_x, point.x);
      predicted_max_y = std::max(predicted_max_y, point.y);
    }
  }
  if (!reference.has_polygon && !predicted_has_polygon) {
    return {};
  }
  if (!predicted_has_polygon) {
    const auto& variant = reference.variants[0];
    ExactMetricCounts output;
    output.gt_area = variant.area;
    output.union_area = variant.area;
    output.recall = output.gt_area > 0 ? 0.0 : 1.0;
    output.precision = 1.0;
    output.iou = output.union_area > 0 ? 0.0 : 1.0;
    return output;
  }

  const int joint_shift_x = reference.has_polygon
      ? static_cast<int>(std::floor(std::min(reference.min_x, predicted_min_x)))
      : static_cast<int>(std::floor(predicted_min_x));
  const int joint_shift_y = reference.has_polygon
      ? static_cast<int>(std::floor(std::min(reference.min_y, predicted_min_y)))
      : static_cast<int>(std::floor(predicted_min_y));
  const int parity_x = positive_modulo_two(joint_shift_x);
  const int parity_y = positive_modulo_two(joint_shift_y);
  const int pred_origin_x = origin_with_parity(
      static_cast<int>(std::floor(predicted_min_x)), parity_x);
  const int pred_origin_y = origin_with_parity(
      static_cast<int>(std::floor(predicted_min_y)), parity_y);
  const int pred_maximum_x = static_cast<int>(std::ceil(predicted_max_x));
  const int pred_maximum_y = static_cast<int>(std::ceil(predicted_max_y));
  const int pred_width = pred_maximum_x - pred_origin_x + 1;
  const int pred_height = pred_maximum_y - pred_origin_y + 1;
  if (pred_width <= 0 || pred_height <= 0) {
    throw std::invalid_argument("polygon bounds produce an invalid raster size");
  }
  const std::size_t pred_pixels =
      static_cast<std::size_t>(pred_width) *
      static_cast<std::size_t>(pred_height);
  if (scratch->prediction.size() < pred_pixels) {
    scratch->prediction.resize(pred_pixels);
  }
  cv::Mat pred_mask(
      pred_height,
      pred_width,
      CV_8UC1,
      scratch->prediction.data(),
      static_cast<std::size_t>(pred_width));
  rasterize_into(
      predicted,
      pred_mask,
      static_cast<float>(pred_origin_x),
      static_cast<float>(pred_origin_y));

  ExactMetricCounts output;
  output.pred_area = cv::countNonZero(pred_mask);
  if (reference.has_polygon) {
    const auto& variant = reference.variants[
        static_cast<std::size_t>(parity_x + 2 * parity_y)];
    output.gt_area = variant.area;
    for (const RasterRun& run : variant.runs) {
      const int local_y = run.y - pred_origin_y;
      if (local_y < 0 || local_y >= pred_height) {
        continue;
      }
      const int start_x = std::max(run.start_x, pred_origin_x);
      const int end_x = std::min(run.end_x, pred_origin_x + pred_width);
      if (start_x >= end_x) {
        continue;
      }
      // A reference row is already compressed into short foreground runs.
      // Calling countNonZero for every run constructs two Mat views and
      // dispatches an OpenCV kernel thousands of times per candidate.  That
      // fixed cost is larger than the work for the typical narrow mask run.
      // Summing the contiguous bytes directly keeps the exact pixel contract
      // and lets -O3 vectorize the reduction without per-run dispatch.
      const std::uint8_t* predicted_row = pred_mask.ptr<std::uint8_t>(local_y);
      const int local_start_x = start_x - pred_origin_x;
      const int local_end_x = end_x - pred_origin_x;
      std::int64_t run_intersection = 0;
      for (int local_x = local_start_x; local_x < local_end_x; ++local_x) {
        run_intersection += predicted_row[local_x];
      }
      output.intersection += run_intersection;
    }
  }
  output.union_area = output.gt_area + output.pred_area - output.intersection;
  output.recall = output.gt_area > 0
      ? static_cast<double>(output.intersection) / output.gt_area
      : 1.0;
  output.precision = output.pred_area > 0
      ? static_cast<double>(output.intersection) / output.pred_area
      : 1.0;
  output.iou = output.union_area > 0
      ? static_cast<double>(output.intersection) / output.union_area
      : 1.0;
  return output;
}

py::array_t<double> pair_vote_local_metrics_batch(
    const py::iterable& gt_values,
    const py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>& chosen_values,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& current_values,
    const int key_pos,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& trial_values,
    const int contour_count,
    const int anchors_per_contour,
    const int threads) {
  if (chosen_values.ndim() != 1 || current_values.ndim() < 2 ||
      trial_values.ndim() < 2) {
    throw py::value_error("invalid pair-vote batch dimensions");
  }
  const auto chosen_view = chosen_values.unchecked<1>();
  const py::ssize_t key_count = chosen_values.shape(0);
  if (key_pos < 0 || key_pos >= key_count || current_values.shape(0) != key_count) {
    throw py::value_error("key_pos/current size does not match chosen frames");
  }
  const std::size_t vector_size = static_cast<std::size_t>(contour_count) *
      static_cast<std::size_t>(anchors_per_contour) * 2U;
  if (vector_size == 0 ||
      static_cast<std::size_t>(current_values.size() / key_count) != vector_size ||
      static_cast<std::size_t>(trial_values.size() / trial_values.shape(0)) != vector_size) {
    throw py::value_error("pair-vote vector size does not match contour layout");
  }
  std::vector<int> chosen(static_cast<std::size_t>(key_count));
  for (py::ssize_t index = 0; index < key_count; ++index) {
    chosen[static_cast<std::size_t>(index)] = chosen_view(index);
  }
  const int left_key = std::max(0, key_pos - 1);
  const int right_key = std::min(static_cast<int>(key_count) - 1, key_pos + 1);
  const int start_frame = chosen[static_cast<std::size_t>(left_key)];
  const int end_frame = chosen[static_cast<std::size_t>(right_key)];
  const py::sequence gt_sequence = py::reinterpret_borrow<py::sequence>(gt_values);
  if (start_frame < 0 || end_frame >= static_cast<int>(gt_sequence.size())) {
    throw py::value_error("chosen frame range exceeds GT sequence");
  }
  std::vector<ExactFrameReference> gt_references;
  gt_references.reserve(static_cast<std::size_t>(end_frame - start_frame + 1));
  for (int frame = start_frame; frame <= end_frame; ++frame) {
    gt_references.push_back(build_exact_frame_reference(parse_polygons(
        py::reinterpret_borrow<py::iterable>(gt_sequence[frame]))));
  }
  const float* current_ptr = static_cast<const float*>(current_values.data());
  const float* trial_ptr = static_cast<const float*>(trial_values.data());
  const py::ssize_t trial_count = trial_values.shape(0);
  py::array_t<double> output({trial_count, static_cast<py::ssize_t>(2)});
  double* output_ptr = static_cast<double*>(output.mutable_data());
  const int thread_count = threads > 0 ? threads : 1;
  std::vector<ExactRasterScratch> thread_scratch(
      static_cast<std::size_t>(thread_count));

  {
    py::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(thread_count)
#endif
    for (py::ssize_t trial_index = 0; trial_index < trial_count; ++trial_index) {
#ifdef _OPENMP
      const int thread_index = omp_get_thread_num();
#else
      const int thread_index = 0;
#endif
      double iou_sum = 0.0;
      double minimum_recall = 1.0;
      const float* trial = trial_ptr + static_cast<std::size_t>(trial_index) * vector_size;
      for (int frame = start_frame; frame <= end_frame; ++frame) {
        const auto right_it = std::lower_bound(chosen.begin(), chosen.end(), frame);
        const int right_pos = static_cast<int>(right_it - chosen.begin());
        const int left_pos = std::max(0, right_pos - 1);
        const bool exact_key = right_pos < static_cast<int>(chosen.size()) &&
            chosen[static_cast<std::size_t>(right_pos)] == frame;
        double alpha64 = 0.0;
        float alpha = 0.0F;
        if (!exact_key) {
          alpha64 =
              static_cast<double>(frame - chosen[static_cast<std::size_t>(left_pos)]) /
              std::max(chosen[static_cast<std::size_t>(right_pos)] -
                           chosen[static_cast<std::size_t>(left_pos)], 1);
          alpha = static_cast<float>(alpha64);
        }
        const auto vector_at = [&](const int position, const std::size_t offset) {
          return position == key_pos
              ? trial[offset]
              : current_ptr[static_cast<std::size_t>(position) * vector_size + offset];
        };
        std::vector<Polygon> pred_polygons(static_cast<std::size_t>(contour_count));
        for (int contour = 0; contour < contour_count; ++contour) {
          auto& polygon = pred_polygons[static_cast<std::size_t>(contour)];
          polygon.reserve(static_cast<std::size_t>(anchors_per_contour));
          for (int point = 0; point < anchors_per_contour; ++point) {
            const std::size_t offset =
                (static_cast<std::size_t>(contour) * anchors_per_contour + point) * 2U;
            float x = 0.0F;
            float y = 0.0F;
            if (exact_key) {
              x = vector_at(right_pos, offset);
              y = vector_at(right_pos, offset + 1U);
            } else {
              // NumPy computes `1.0 - float(alpha)` in Python float64 and
              // casts that scalar for the float32 array multiplication.
              const float beta = static_cast<float>(1.0 - alpha64);
              x = beta * vector_at(left_pos, offset) +
                  alpha * vector_at(right_pos, offset);
              y = beta * vector_at(left_pos, offset + 1U) +
                  alpha * vector_at(right_pos, offset + 1U);
            }
            polygon.emplace_back(x, y);
          }
        }
        const ExactMetricCounts metrics = exact_metric_counts_from_reference(
            gt_references[static_cast<std::size_t>(frame - start_frame)],
            pred_polygons,
            &thread_scratch[static_cast<std::size_t>(thread_index)]);
        iou_sum += metrics.iou;
        minimum_recall = std::min(minimum_recall, metrics.recall);
      }
      output_ptr[trial_index * 2] = iou_sum;
      output_ptr[trial_index * 2 + 1] = minimum_recall;
    }
  }
  return output;
}

py::array_t<double> pair_vote_full_metrics_batch(
    const py::iterable& gt_values,
    const py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>& chosen_values,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& trial_values,
    const int contour_count,
    const int anchors_per_contour,
    const int threads) {
  if (chosen_values.ndim() != 1 || trial_values.ndim() < 3) {
    throw py::value_error("invalid full pair-vote batch dimensions");
  }
  const py::ssize_t key_count = chosen_values.shape(0);
  const py::ssize_t trial_count = trial_values.shape(0);
  if (key_count <= 0 || trial_count <= 0 || trial_values.shape(1) != key_count) {
    throw py::value_error("full pair-vote batch key dimensions do not match");
  }
  const std::size_t vector_size = static_cast<std::size_t>(contour_count) *
      static_cast<std::size_t>(anchors_per_contour) * 2U;
  if (vector_size == 0 ||
      static_cast<std::size_t>(trial_values.size() / (trial_count * key_count)) !=
          vector_size) {
    throw py::value_error("full pair-vote vector size does not match contour layout");
  }
  const auto chosen_view = chosen_values.unchecked<1>();
  std::vector<int> chosen(static_cast<std::size_t>(key_count));
  for (py::ssize_t index = 0; index < key_count; ++index) {
    chosen[static_cast<std::size_t>(index)] = chosen_view(index);
  }
  const py::sequence gt_sequence = py::reinterpret_borrow<py::sequence>(gt_values);
  const int frame_count = static_cast<int>(gt_sequence.size());
  std::vector<ExactFrameReference> gt_references;
  gt_references.reserve(static_cast<std::size_t>(frame_count));
  for (int frame = 0; frame < frame_count; ++frame) {
    gt_references.push_back(build_exact_frame_reference(parse_polygons(
        py::reinterpret_borrow<py::iterable>(gt_sequence[frame]))));
  }
  const float* trial_ptr = static_cast<const float*>(trial_values.data());
  py::array_t<double> output({trial_count, static_cast<py::ssize_t>(2)});
  double* output_ptr = static_cast<double*>(output.mutable_data());
  const int thread_count = threads > 0 ? threads : 1;
  std::vector<ExactRasterScratch> thread_scratch(
      static_cast<std::size_t>(thread_count));

  {
    py::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(thread_count)
#endif
    for (py::ssize_t trial_index = 0; trial_index < trial_count; ++trial_index) {
#ifdef _OPENMP
      const int thread_index = omp_get_thread_num();
#else
      const int thread_index = 0;
#endif
      const float* keys = trial_ptr +
          static_cast<std::size_t>(trial_index) *
              static_cast<std::size_t>(key_count) * vector_size;
      double total_iou_loss = 0.0;
      double minimum_recall = 1.0;
      for (int frame = 0; frame < frame_count; ++frame) {
        int left_pos = 0;
        int right_pos = 0;
        bool exact_key = true;
        double alpha64 = 0.0;
        if (frame <= chosen.front()) {
          left_pos = right_pos = 0;
        } else if (frame >= chosen.back()) {
          left_pos = right_pos = static_cast<int>(key_count) - 1;
        } else {
          const auto right_it = std::lower_bound(chosen.begin(), chosen.end(), frame);
          right_pos = static_cast<int>(right_it - chosen.begin());
          if (chosen[static_cast<std::size_t>(right_pos)] == frame) {
            left_pos = right_pos;
          } else {
            exact_key = false;
            left_pos = std::max(0, right_pos - 1);
            alpha64 =
                static_cast<double>(frame - chosen[static_cast<std::size_t>(left_pos)]) /
                std::max(chosen[static_cast<std::size_t>(right_pos)] -
                             chosen[static_cast<std::size_t>(left_pos)], 1);
          }
        }
        const float alpha = static_cast<float>(alpha64);
        const float beta = static_cast<float>(1.0 - alpha64);
        std::vector<Polygon> pred_polygons(static_cast<std::size_t>(contour_count));
        for (int contour = 0; contour < contour_count; ++contour) {
          auto& polygon = pred_polygons[static_cast<std::size_t>(contour)];
          polygon.reserve(static_cast<std::size_t>(anchors_per_contour));
          for (int point = 0; point < anchors_per_contour; ++point) {
            const std::size_t offset =
                (static_cast<std::size_t>(contour) * anchors_per_contour + point) * 2U;
            const float* left = keys + static_cast<std::size_t>(left_pos) * vector_size;
            const float* right = keys + static_cast<std::size_t>(right_pos) * vector_size;
            const float x = exact_key
                ? right[offset]
                : beta * left[offset] + alpha * right[offset];
            const float y = exact_key
                ? right[offset + 1U]
                : beta * left[offset + 1U] + alpha * right[offset + 1U];
            polygon.emplace_back(x, y);
          }
        }
        const ExactMetricCounts metrics = exact_metric_counts_from_reference(
            gt_references[static_cast<std::size_t>(frame)],
            pred_polygons,
            &thread_scratch[static_cast<std::size_t>(thread_index)]);
        total_iou_loss += 1.0 - metrics.iou;
        minimum_recall = std::min(minimum_recall, metrics.recall);
      }
      output_ptr[trial_index * 2] =
          1.0 - total_iou_loss / static_cast<double>(std::max(frame_count, 1));
      output_ptr[trial_index * 2 + 1] = minimum_recall;
    }
  }
  return output;
}

py::dict exact_metrics(const py::iterable& gt_values, const py::iterable& pred_values) {
  const auto gt_polygons = parse_polygons(gt_values);
  const auto pred_polygons = parse_polygons(pred_values);

  bool has_valid_polygon = false;
  float min_x = std::numeric_limits<float>::infinity();
  float min_y = std::numeric_limits<float>::infinity();
  float max_x = -std::numeric_limits<float>::infinity();
  float max_y = -std::numeric_limits<float>::infinity();

  const auto include_bounds = [&](const std::vector<Polygon>& polygons) {
    for (const auto& polygon : polygons) {
      if (polygon.size() < 3) {
        continue;
      }
      has_valid_polygon = true;
      for (const auto& point : polygon) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
          throw py::value_error("polygon coordinates must be finite");
        }
        min_x = std::min(min_x, point.x);
        min_y = std::min(min_y, point.y);
        max_x = std::max(max_x, point.x);
        max_y = std::max(max_y, point.y);
      }
    }
  };
  include_bounds(gt_polygons);
  include_bounds(pred_polygons);

  py::dict result;
  if (!has_valid_polygon) {
    result["gt_area"] = 0.0;
    result["pred_area"] = 0.0;
    result["intersection"] = 0.0;
    result["union"] = 0.0;
    result["recall"] = 1.0;
    result["precision"] = 1.0;
    result["iou"] = 1.0;
    return result;
  }

  const int min_x_i = static_cast<int>(std::floor(min_x));
  const int min_y_i = static_cast<int>(std::floor(min_y));
  const int max_x_i = static_cast<int>(std::ceil(max_x));
  const int max_y_i = static_cast<int>(std::ceil(max_y));
  const std::int64_t width64 = static_cast<std::int64_t>(max_x_i) - min_x_i + 1;
  const std::int64_t height64 = static_cast<std::int64_t>(max_y_i) - min_y_i + 1;
  if (width64 <= 0 || height64 <= 0 ||
      width64 > std::numeric_limits<int>::max() ||
      height64 > std::numeric_limits<int>::max()) {
    throw py::value_error("polygon bounds produce an invalid raster size");
  }

  const cv::Mat gt_mask = rasterize(
      gt_polygons, static_cast<int>(height64), static_cast<int>(width64),
      static_cast<float>(min_x_i), static_cast<float>(min_y_i));
  const cv::Mat pred_mask = rasterize(
      pred_polygons, static_cast<int>(height64), static_cast<int>(width64),
      static_cast<float>(min_x_i), static_cast<float>(min_y_i));

  const std::int64_t gt_area = cv::countNonZero(gt_mask);
  const std::int64_t pred_area = cv::countNonZero(pred_mask);
  cv::Mat intersection_mask;
  cv::bitwise_and(gt_mask, pred_mask, intersection_mask);
  const std::int64_t intersection = cv::countNonZero(intersection_mask);
  const std::int64_t union_area = gt_area + pred_area - intersection;

  result["gt_area"] = static_cast<double>(gt_area);
  result["pred_area"] = static_cast<double>(pred_area);
  result["intersection"] = static_cast<double>(intersection);
  result["union"] = static_cast<double>(union_area);
  result["recall"] = gt_area > 0 ? static_cast<double>(intersection) / gt_area : 1.0;
  result["precision"] = pred_area > 0 ? static_cast<double>(intersection) / pred_area : 1.0;
  result["iou"] = union_area > 0 ? static_cast<double>(intersection) / union_area : 1.0;
  return result;
}

using RoundedPolygon = std::vector<cv::Point>;

std::vector<RoundedPolygon> round_polygons_globally(
    const std::vector<Polygon>& polygons) {
  std::vector<RoundedPolygon> output;
  output.reserve(polygons.size());
  for (const auto& polygon : polygons) {
    RoundedPolygon rounded;
    rounded.reserve(polygon.size());
    for (const auto& point : polygon) {
      if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
        throw py::value_error("polygon coordinates must be finite");
      }
      // Round in the global coordinate system before choosing a compact ROI.
      // This makes the GT raster invariant to the prediction bounds.
      rounded.emplace_back(
          static_cast<int>(std::nearbyint(point.x)),
          static_cast<int>(std::nearbyint(point.y)));
    }
    output.push_back(std::move(rounded));
  }
  return output;
}

cv::Mat rasterize_globally_rounded(
    const std::vector<RoundedPolygon>& polygons,
    const int height,
    const int width,
    const int shift_x,
    const int shift_y) {
  cv::Mat mask = cv::Mat::zeros(height, width, CV_8UC1);
  for (const auto& polygon : polygons) {
    if (polygon.size() < 3) {
      continue;
    }
    RoundedPolygon local;
    local.reserve(polygon.size());
    for (const auto& point : polygon) {
      local.emplace_back(point.x - shift_x, point.y - shift_y);
    }
    const std::vector<RoundedPolygon> one_polygon{std::move(local)};
    cv::fillPoly(mask, one_polygon, cv::Scalar(1));
  }
  return mask;
}

py::dict canonical_metrics(
    const py::iterable& gt_values,
    const py::iterable& pred_values) {
  const auto gt = round_polygons_globally(parse_polygons(gt_values));
  const auto pred = round_polygons_globally(parse_polygons(pred_values));
  bool has_valid_polygon = false;
  int min_x = std::numeric_limits<int>::max();
  int min_y = std::numeric_limits<int>::max();
  int max_x = std::numeric_limits<int>::min();
  int max_y = std::numeric_limits<int>::min();
  const auto include_bounds = [&](const std::vector<RoundedPolygon>& polygons) {
    for (const auto& polygon : polygons) {
      if (polygon.size() < 3) {
        continue;
      }
      has_valid_polygon = true;
      for (const auto& point : polygon) {
        min_x = std::min(min_x, point.x);
        min_y = std::min(min_y, point.y);
        max_x = std::max(max_x, point.x);
        max_y = std::max(max_y, point.y);
      }
    }
  };
  include_bounds(gt);
  include_bounds(pred);

  py::dict result;
  if (!has_valid_polygon) {
    result["gt_area"] = 0.0;
    result["pred_area"] = 0.0;
    result["intersection"] = 0.0;
    result["union"] = 0.0;
    result["recall"] = 1.0;
    result["precision"] = 1.0;
    result["iou"] = 1.0;
    return result;
  }
  const std::int64_t width64 = static_cast<std::int64_t>(max_x) - min_x + 1;
  const std::int64_t height64 = static_cast<std::int64_t>(max_y) - min_y + 1;
  if (width64 <= 0 || height64 <= 0 ||
      width64 > std::numeric_limits<int>::max() ||
      height64 > std::numeric_limits<int>::max()) {
    throw py::value_error("rounded polygon bounds produce an invalid raster size");
  }
  const cv::Mat gt_mask = rasterize_globally_rounded(
      gt, static_cast<int>(height64), static_cast<int>(width64), min_x, min_y);
  const cv::Mat pred_mask = rasterize_globally_rounded(
      pred, static_cast<int>(height64), static_cast<int>(width64), min_x, min_y);
  const std::int64_t gt_area = cv::countNonZero(gt_mask);
  const std::int64_t pred_area = cv::countNonZero(pred_mask);
  cv::Mat intersection_mask;
  cv::bitwise_and(gt_mask, pred_mask, intersection_mask);
  const std::int64_t intersection = cv::countNonZero(intersection_mask);
  const std::int64_t union_area = gt_area + pred_area - intersection;
  result["gt_area"] = static_cast<double>(gt_area);
  result["pred_area"] = static_cast<double>(pred_area);
  result["intersection"] = static_cast<double>(intersection);
  result["union"] = static_cast<double>(union_area);
  result["recall"] = gt_area > 0 ? static_cast<double>(intersection) / gt_area : 1.0;
  result["precision"] = pred_area > 0 ? static_cast<double>(intersection) / pred_area : 1.0;
  result["iou"] = union_area > 0 ? static_cast<double>(intersection) / union_area : 1.0;
  return result;
}

struct DoubleRasterScratch {
  cv::Mat reference;
  cv::Mat prediction;
  cv::Mat intersection;
  std::vector<std::uint8_t> reference_storage;
  std::vector<std::uint8_t> prediction_storage;
  std::vector<std::uint8_t> intersection_storage;
  RoundedPolygon rounded_storage;
};

using DoublePolygon = std::vector<cv::Point2d>;

struct DoubleReferenceMaskVariant {
  cv::Mat bitmap;
  int origin_x = 0;
  int origin_y = 0;
  std::vector<RasterRun> runs;
  std::int64_t area = 0;
};

struct DoubleFrameReference {
  std::vector<DoublePolygon> polygons;
  bool has_polygon = false;
  bool masks_cached = false;
  bool bitmaps_cached = false;
  double min_x = std::numeric_limits<double>::infinity();
  double min_y = std::numeric_limits<double>::infinity();
  double max_x = -std::numeric_limits<double>::infinity();
  double max_y = -std::numeric_limits<double>::infinity();
  std::array<DoubleReferenceMaskVariant, 4> variants;
};

std::vector<DoublePolygon> parse_polygons64(const py::iterable& values) {
  std::vector<DoublePolygon> output;
  for (const py::handle value : values) {
    const auto array =
        py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(value);
    if (!array || array.ndim() != 2 || array.shape(1) != 2) {
      throw py::value_error("each polygon must have shape (N, 2)");
    }
    const auto view = array.unchecked<2>();
    DoublePolygon polygon;
    polygon.reserve(static_cast<std::size_t>(array.shape(0)));
    for (py::ssize_t index = 0; index < array.shape(0); ++index) {
      const double x = view(index, 0);
      const double y = view(index, 1);
      if (!std::isfinite(x) || !std::isfinite(y)) {
        throw py::value_error("polygon coordinates must be finite");
      }
      polygon.emplace_back(x, y);
    }
    output.push_back(std::move(polygon));
  }
  return output;
}

void rasterize_double_into(
    const std::vector<DoublePolygon>& polygons,
    cv::Mat* mask,
    const int height,
    const int width,
    const int shift_x,
    const int shift_y) {
  mask->create(height, width, CV_8UC1);
  mask->setTo(cv::Scalar(0));
  for (const auto& polygon : polygons) {
    if (polygon.size() < 3) {
      continue;
    }
    RoundedPolygon local;
    local.reserve(polygon.size());
    for (const auto& point : polygon) {
      local.emplace_back(
          static_cast<int>(std::nearbyint(point.x - shift_x)),
          static_cast<int>(std::nearbyint(point.y - shift_y)));
    }
    const std::vector<RoundedPolygon> one_polygon{std::move(local)};
    cv::fillPoly(*mask, one_polygon, cv::Scalar(1));
  }
}

void rasterize_double_buffered_into(
    const std::vector<DoublePolygon>& polygons,
    cv::Mat* mask,
    std::vector<std::uint8_t>* storage,
    RoundedPolygon* rounded_storage,
    const int height,
    const int width,
    const int shift_x,
    const int shift_y) {
  const std::size_t pixels =
      static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
  if (storage->size() < pixels) {
    storage->resize(pixels);
  }
  *mask = cv::Mat(
      height,
      width,
      CV_8UC1,
      storage->data(),
      static_cast<std::size_t>(width));
  mask->setTo(cv::Scalar(0));
  for (const auto& polygon : polygons) {
    if (polygon.size() < 3) {
      continue;
    }
    rounded_storage->clear();
    rounded_storage->reserve(polygon.size());
    for (const auto& point : polygon) {
      rounded_storage->emplace_back(
          static_cast<int>(std::nearbyint(point.x - shift_x)),
          static_cast<int>(std::nearbyint(point.y - shift_y)));
    }
    const cv::Point* points = rounded_storage->data();
    const int point_count = static_cast<int>(rounded_storage->size());
    cv::fillPoly(*mask, &points, &point_count, 1, cv::Scalar(1));
  }
}

cv::Mat buffered_mask(
    std::vector<std::uint8_t>* storage,
    const int height,
    const int width) {
  const std::size_t pixels =
      static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
  if (storage->size() < pixels) {
    storage->resize(pixels);
  }
  return cv::Mat(
      height,
      width,
      CV_8UC1,
      storage->data(),
      static_cast<std::size_t>(width));
}

DoubleFrameReference build_double_frame_reference(
    std::vector<DoublePolygon> polygons,
    const bool cache_masks) {
  DoubleFrameReference output;
  output.polygons = std::move(polygons);
  for (const auto& polygon : output.polygons) {
    if (polygon.size() < 3) {
      continue;
    }
    output.has_polygon = true;
    for (const auto& point : polygon) {
      output.min_x = std::min(output.min_x, point.x);
      output.min_y = std::min(output.min_y, point.y);
      output.max_x = std::max(output.max_x, point.x);
      output.max_y = std::max(output.max_y, point.y);
    }
  }
  if (!output.has_polygon || !cache_masks) {
    return output;
  }
  const int floor_x = static_cast<int>(std::floor(output.min_x));
  const int floor_y = static_cast<int>(std::floor(output.min_y));
  const int maximum_x = static_cast<int>(std::ceil(output.max_x));
  const int maximum_y = static_cast<int>(std::ceil(output.max_y));
  for (int parity_y = 0; parity_y < 2; ++parity_y) {
    for (int parity_x = 0; parity_x < 2; ++parity_x) {
      const int index = parity_x + 2 * parity_y;
      auto& variant = output.variants[static_cast<std::size_t>(index)];
      const int origin_x = origin_with_parity(floor_x, parity_x);
      const int origin_y = origin_with_parity(floor_y, parity_y);
      const int width = maximum_x - origin_x + 1;
      const int height = maximum_y - origin_y + 1;
      cv::Mat mask;
      rasterize_double_into(
          output.polygons,
          &mask,
          height,
          width,
          origin_x,
          origin_y);
      variant.origin_x = origin_x;
      variant.origin_y = origin_y;
      for (int y = 0; y < height; ++y) {
        const std::uint8_t* row = mask.ptr<std::uint8_t>(y);
        int x = 0;
        while (x < width) {
          while (x < width && row[x] == 0) {
            ++x;
          }
          if (x >= width) {
            break;
          }
          const int start = x;
          while (x < width && row[x] != 0) {
            ++x;
          }
          variant.runs.push_back(
              {y + origin_y, start + origin_x, x + origin_x});
          variant.area += static_cast<std::int64_t>(x - start);
        }
      }
    }
  }
  output.masks_cached = true;
  return output;
}

std::uint64_t double_reference_run_bytes(
    const DoubleFrameReference& reference) {
  std::uint64_t total = 0U;
  for (const auto& variant : reference.variants) {
    total += static_cast<std::uint64_t>(variant.runs.size()) *
        static_cast<std::uint64_t>(sizeof(RasterRun));
  }
  return total;
}

std::uint64_t double_reference_bitmap_bytes(
    const DoubleFrameReference& reference) {
  if (!reference.has_polygon) {
    return 0U;
  }
  const int floor_x = static_cast<int>(std::floor(reference.min_x));
  const int floor_y = static_cast<int>(std::floor(reference.min_y));
  const int maximum_x = static_cast<int>(std::ceil(reference.max_x));
  const int maximum_y = static_cast<int>(std::ceil(reference.max_y));
  std::uint64_t total = 0U;
  for (int parity_y = 0; parity_y < 2; ++parity_y) {
    for (int parity_x = 0; parity_x < 2; ++parity_x) {
      const int origin_x = origin_with_parity(floor_x, parity_x);
      const int origin_y = origin_with_parity(floor_y, parity_y);
      total += static_cast<std::uint64_t>(maximum_x - origin_x + 1) *
          static_cast<std::uint64_t>(maximum_y - origin_y + 1);
    }
  }
  return total;
}

void cache_double_reference_bitmaps(DoubleFrameReference* reference) {
  if (!reference->has_polygon) {
    reference->bitmaps_cached = true;
    return;
  }
  const int floor_x = static_cast<int>(std::floor(reference->min_x));
  const int floor_y = static_cast<int>(std::floor(reference->min_y));
  const int maximum_x = static_cast<int>(std::ceil(reference->max_x));
  const int maximum_y = static_cast<int>(std::ceil(reference->max_y));
  for (int parity_y = 0; parity_y < 2; ++parity_y) {
    for (int parity_x = 0; parity_x < 2; ++parity_x) {
      const int index = parity_x + 2 * parity_y;
      auto& variant = reference->variants[static_cast<std::size_t>(index)];
      variant.origin_x = origin_with_parity(floor_x, parity_x);
      variant.origin_y = origin_with_parity(floor_y, parity_y);
      rasterize_double_into(
          reference->polygons,
          &variant.bitmap,
          maximum_y - variant.origin_y + 1,
          maximum_x - variant.origin_x + 1,
          variant.origin_x,
          variant.origin_y);
      variant.area = cv::countNonZero(variant.bitmap);
      variant.runs.clear();
      variant.runs.shrink_to_fit();
    }
  }
  reference->masks_cached = true;
  reference->bitmaps_cached = true;
}

ExactMetricCounts double_metric_counts(
    const std::vector<DoublePolygon>& reference,
    const std::vector<DoublePolygon>& predicted,
    DoubleRasterScratch* scratch) {
  bool has_polygon = false;
  double minimum_x = std::numeric_limits<double>::infinity();
  double minimum_y = std::numeric_limits<double>::infinity();
  double maximum_x = -std::numeric_limits<double>::infinity();
  double maximum_y = -std::numeric_limits<double>::infinity();
  const auto include = [&](const std::vector<DoublePolygon>& polygons) {
    for (const auto& polygon : polygons) {
      if (polygon.size() < 3) {
        continue;
      }
      has_polygon = true;
      for (const auto& point : polygon) {
        minimum_x = std::min(minimum_x, point.x);
        minimum_y = std::min(minimum_y, point.y);
        maximum_x = std::max(maximum_x, point.x);
        maximum_y = std::max(maximum_y, point.y);
      }
    }
  };
  include(reference);
  include(predicted);
  if (!has_polygon) {
    return {};
  }
  const int shift_x = static_cast<int>(std::floor(minimum_x)) - 2;
  const int shift_y = static_cast<int>(std::floor(minimum_y)) - 2;
  const int maximum_x_i = static_cast<int>(std::ceil(maximum_x)) + 2;
  const int maximum_y_i = static_cast<int>(std::ceil(maximum_y)) + 2;
  const std::int64_t width64 =
      static_cast<std::int64_t>(maximum_x_i) - shift_x + 1;
  const std::int64_t height64 =
      static_cast<std::int64_t>(maximum_y_i) - shift_y + 1;
  if (width64 <= 0 || height64 <= 0 ||
      width64 > std::numeric_limits<int>::max() ||
      height64 > std::numeric_limits<int>::max()) {
    throw std::invalid_argument("double polygon bounds produce an invalid raster size");
  }
  rasterize_double_buffered_into(
      reference,
      &scratch->reference,
      &scratch->reference_storage,
      &scratch->rounded_storage,
      static_cast<int>(height64),
      static_cast<int>(width64),
      shift_x,
      shift_y);
  rasterize_double_buffered_into(
      predicted,
      &scratch->prediction,
      &scratch->prediction_storage,
      &scratch->rounded_storage,
      static_cast<int>(height64),
      static_cast<int>(width64),
      shift_x,
      shift_y);
  scratch->intersection = buffered_mask(
      &scratch->intersection_storage,
      static_cast<int>(height64),
      static_cast<int>(width64));
  cv::bitwise_and(
      scratch->reference,
      scratch->prediction,
      scratch->intersection);
  ExactMetricCounts output;
  output.gt_area = cv::countNonZero(scratch->reference);
  output.pred_area = cv::countNonZero(scratch->prediction);
  output.intersection = cv::countNonZero(scratch->intersection);
  output.union_area = output.gt_area + output.pred_area - output.intersection;
  output.recall = output.gt_area > 0
      ? static_cast<double>(output.intersection) / output.gt_area
      : 1.0;
  output.precision = output.pred_area > 0
      ? static_cast<double>(output.intersection) / output.pred_area
      : 1.0;
  output.iou = output.union_area > 0
      ? static_cast<double>(output.intersection) / output.union_area
      : 1.0;
  return output;
}

ExactMetricCounts cached_double_metric_counts(
    const DoubleFrameReference& reference,
    const std::vector<DoublePolygon>& predicted,
    DoubleRasterScratch* scratch) {
  if (!reference.masks_cached) {
    return double_metric_counts(reference.polygons, predicted, scratch);
  }
  bool predicted_has_polygon = false;
  double predicted_min_x = std::numeric_limits<double>::infinity();
  double predicted_min_y = std::numeric_limits<double>::infinity();
  double predicted_max_x = -std::numeric_limits<double>::infinity();
  double predicted_max_y = -std::numeric_limits<double>::infinity();
  for (const auto& polygon : predicted) {
    if (polygon.size() < 3) {
      continue;
    }
    predicted_has_polygon = true;
    for (const auto& point : polygon) {
      if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
        throw std::invalid_argument("polygon coordinates must be finite");
      }
      predicted_min_x = std::min(predicted_min_x, point.x);
      predicted_min_y = std::min(predicted_min_y, point.y);
      predicted_max_x = std::max(predicted_max_x, point.x);
      predicted_max_y = std::max(predicted_max_y, point.y);
    }
  }
  if (!reference.has_polygon && !predicted_has_polygon) {
    return {};
  }
  if (!predicted_has_polygon) {
    const auto& variant = reference.variants[0];
    ExactMetricCounts output;
    output.gt_area = variant.area;
    output.union_area = variant.area;
    output.recall = output.gt_area > 0 ? 0.0 : 1.0;
    output.precision = 1.0;
    output.iou = output.union_area > 0 ? 0.0 : 1.0;
    return output;
  }

  const int joint_shift_x = reference.has_polygon
      ? static_cast<int>(std::floor(std::min(reference.min_x, predicted_min_x)))
      : static_cast<int>(std::floor(predicted_min_x));
  const int joint_shift_y = reference.has_polygon
      ? static_cast<int>(std::floor(std::min(reference.min_y, predicted_min_y)))
      : static_cast<int>(std::floor(predicted_min_y));
  const int parity_x = positive_modulo_two(joint_shift_x);
  const int parity_y = positive_modulo_two(joint_shift_y);
  const int pred_origin_x = origin_with_parity(
      static_cast<int>(std::floor(predicted_min_x)), parity_x);
  const int pred_origin_y = origin_with_parity(
      static_cast<int>(std::floor(predicted_min_y)), parity_y);
  const int pred_maximum_x = static_cast<int>(std::ceil(predicted_max_x));
  const int pred_maximum_y = static_cast<int>(std::ceil(predicted_max_y));
  const std::int64_t pred_width64 =
      static_cast<std::int64_t>(pred_maximum_x) - pred_origin_x + 1;
  const std::int64_t pred_height64 =
      static_cast<std::int64_t>(pred_maximum_y) - pred_origin_y + 1;
  if (pred_width64 <= 0 || pred_height64 <= 0 ||
      pred_width64 > std::numeric_limits<int>::max() ||
      pred_height64 > std::numeric_limits<int>::max()) {
    throw std::invalid_argument("double polygon bounds produce an invalid raster size");
  }
  const int pred_width = static_cast<int>(pred_width64);
  const int pred_height = static_cast<int>(pred_height64);
  rasterize_double_buffered_into(
      predicted,
      &scratch->prediction,
      &scratch->prediction_storage,
      &scratch->rounded_storage,
      pred_height,
      pred_width,
      pred_origin_x,
      pred_origin_y);

  ExactMetricCounts output;
  output.pred_area = cv::countNonZero(scratch->prediction);
  if (reference.has_polygon) {
    const auto& variant = reference.variants[static_cast<std::size_t>(
        parity_x + 2 * parity_y)];
    output.gt_area = variant.area;
    if (reference.bitmaps_cached) {
      const int start_x = std::max(variant.origin_x, pred_origin_x);
      const int start_y = std::max(variant.origin_y, pred_origin_y);
      const int end_x = std::min(
          variant.origin_x + variant.bitmap.cols,
          pred_origin_x + pred_width);
      const int end_y = std::min(
          variant.origin_y + variant.bitmap.rows,
          pred_origin_y + pred_height);
      if (start_x < end_x && start_y < end_y) {
        const cv::Rect reference_roi(
            start_x - variant.origin_x,
            start_y - variant.origin_y,
            end_x - start_x,
            end_y - start_y);
        const cv::Rect prediction_roi(
            start_x - pred_origin_x,
            start_y - pred_origin_y,
            end_x - start_x,
            end_y - start_y);
        scratch->intersection = buffered_mask(
            &scratch->intersection_storage,
            end_y - start_y,
            end_x - start_x);
        cv::bitwise_and(
            variant.bitmap(reference_roi),
            scratch->prediction(prediction_roi),
            scratch->intersection);
        output.intersection = cv::countNonZero(scratch->intersection);
      }
    } else {
      for (const RasterRun& run : variant.runs) {
        const int local_y = run.y - pred_origin_y;
        if (local_y < 0 || local_y >= pred_height) {
          continue;
        }
        const int start_x = std::max(run.start_x, pred_origin_x);
        const int end_x = std::min(run.end_x, pred_origin_x + pred_width);
        if (start_x >= end_x) {
          continue;
        }
        output.intersection += static_cast<std::int64_t>(cv::countNonZero(
            scratch->prediction.row(local_y).colRange(
                start_x - pred_origin_x,
                end_x - pred_origin_x)));
      }
    }
  }
  output.union_area = output.gt_area + output.pred_area - output.intersection;
  output.recall = output.gt_area > 0
      ? static_cast<double>(output.intersection) / output.gt_area
      : 1.0;
  output.precision = output.pred_area > 0
      ? static_cast<double>(output.intersection) / output.pred_area
      : 1.0;
  output.iou = output.union_area > 0
      ? static_cast<double>(output.intersection) / output.union_area
      : 1.0;
  return output;
}

class ExactDoubleRasterEvaluator {
 public:
  explicit ExactDoubleRasterEvaluator(
      const py::iterable& frame_values,
      const std::uint64_t maximum_cache_bytes) {
    const py::list frames(frame_values);
    std::vector<std::uint64_t> run_costs;
    std::vector<std::uint64_t> bitmap_costs;
    references_.reserve(frames.size());
    run_costs.reserve(frames.size());
    bitmap_costs.reserve(frames.size());
    std::uint64_t total_run_bytes = 0U;
    std::uint64_t total_bitmap_bytes = 0U;
    for (const py::handle frame : frames) {
      references_.push_back(build_double_frame_reference(
          parse_polygons64(py::reinterpret_borrow<py::iterable>(frame)),
          true));
      const std::uint64_t run_bytes =
          double_reference_run_bytes(references_.back());
      const std::uint64_t bitmap_bytes =
          double_reference_bitmap_bytes(references_.back());
      run_costs.push_back(run_bytes);
      bitmap_costs.push_back(bitmap_bytes);
      total_run_bytes += run_bytes;
      total_bitmap_bytes += bitmap_bytes;
    }
    // Four parity variants preserve NumPy ties-to-even exactly.  Every frame
    // first gets a compact row-run representation; within the same hard byte
    // budget, expensive frames are promoted to OpenCV bitmaps.  Bitmap
    // bitwise intersections are materially faster on large/noisy masks while
    // row runs avoid an all-or-nothing cache cliff.  Both encode exactly the
    // same rasterized pixels.
    const std::uint64_t cache_budget_bytes = maximum_cache_bytes;
    std::vector<bool> run_frame(references_.size(), false);
    std::uint64_t used = 0U;
    if (total_run_bytes <= cache_budget_bytes) {
      std::fill(run_frame.begin(), run_frame.end(), true);
      used = total_run_bytes;
    } else {
      std::vector<std::size_t> by_cost(references_.size());
      std::iota(by_cost.begin(), by_cost.end(), 0U);
      std::stable_sort(
          by_cost.begin(), by_cost.end(),
          [&run_costs](const std::size_t left, const std::size_t right) {
            return run_costs[left] < run_costs[right];
          });
      for (const std::size_t index : by_cost) {
        const std::uint64_t cost = run_costs[index];
        if (cost > cache_budget_bytes - used) {
          continue;
        }
        run_frame[index] = true;
        used += cost;
      }
    }
    for (std::size_t index = 0; index < references_.size(); ++index) {
      references_[index].masks_cached = run_frame[index];
      if (run_frame[index]) {
        ++cached_reference_frames_;
        ++run_cached_reference_frames_;
      } else {
        for (auto& variant : references_[index].variants) {
          variant.runs.clear();
          variant.runs.shrink_to_fit();
          variant.area = 0;
        }
      }
    }
    // Promote the largest expensive masks first.  Small masks remain fast in
    // row-run form, while this ordering prevents a few full-screen frames from
    // dominating exact graph evaluation merely because they were too large to
    // win a cheapest-frame cache policy.
    std::vector<std::size_t> bitmap_order(references_.size());
    std::iota(bitmap_order.begin(), bitmap_order.end(), 0U);
    std::stable_sort(
        bitmap_order.begin(), bitmap_order.end(),
        [&bitmap_costs](const std::size_t left, const std::size_t right) {
          return bitmap_costs[left] > bitmap_costs[right];
        });
    for (const std::size_t index : bitmap_order) {
      if (!run_frame[index] || bitmap_costs[index] == 0U) {
        continue;
      }
      const std::uint64_t run_cost = run_costs[index];
      const std::uint64_t available = cache_budget_bytes - used + run_cost;
      if (bitmap_costs[index] > available) {
        continue;
      }
      used = used - run_cost + bitmap_costs[index];
      cache_double_reference_bitmaps(&references_[index]);
      ++bitmap_cached_reference_frames_;
      --run_cached_reference_frames_;
    }
    cached_reference_bytes_ = used;
    total_reference_cache_bytes_ = total_run_bytes;
    total_bitmap_reference_bytes_ = total_bitmap_bytes;
    cache_budget_bytes_ = cache_budget_bytes;
  }

  py::dict cache_stats() const {
    py::dict output;
    output["reference_frames"] = references_.size();
    output["cached_reference_frames"] = cached_reference_frames_;
    output["uncached_reference_frames"] =
        references_.size() - cached_reference_frames_;
    output["cached_reference_bytes"] = cached_reference_bytes_;
    output["total_reference_cache_bytes"] = total_reference_cache_bytes_;
    output["run_cached_reference_frames"] = run_cached_reference_frames_;
    output["bitmap_cached_reference_frames"] =
        bitmap_cached_reference_frames_;
    output["total_bitmap_reference_bytes"] = total_bitmap_reference_bytes_;
    output["cache_budget_bytes"] = cache_budget_bytes_;
    return output;
  }

  py::array_t<double> metrics_batch(
      const py::array_t<
          std::int32_t,
          py::array::c_style | py::array::forcecast>& frame_indices,
      const py::array_t<
          double,
          py::array::c_style | py::array::forcecast>& vectors,
      const int contour_count,
      const int anchors_per_contour,
      const int requested_threads) const {
    if (frame_indices.ndim() != 1) {
      throw py::value_error("frame_indices must have shape (N,)");
    }
    if (vectors.ndim() != 3 || vectors.shape(2) != 2 ||
        vectors.shape(0) != frame_indices.shape(0)) {
      throw py::value_error("vectors must have shape (N, points, 2)");
    }
    const py::ssize_t expected_points =
        static_cast<py::ssize_t>(contour_count) * anchors_per_contour;
    if (contour_count < 0 || anchors_per_contour < 0 ||
        vectors.shape(1) != expected_points) {
      throw py::value_error("vector point count does not match contour topology");
    }
    const auto frame_view = frame_indices.unchecked<1>();
    const py::ssize_t case_count = frame_indices.shape(0);
    for (py::ssize_t index = 0; index < case_count; ++index) {
      if (frame_view(index) < 0 ||
          frame_view(index) >= static_cast<int>(references_.size())) {
        throw py::value_error("frame index is outside the evaluator sequence");
      }
    }
    py::array_t<double> output({case_count, static_cast<py::ssize_t>(7)});
    auto output_view = output.mutable_unchecked<2>();
    const double* input = vectors.data();
    const std::size_t values_per_case =
        static_cast<std::size_t>(expected_points) * 2U;
    const int thread_count = std::max(1, requested_threads);
    std::lock_guard<std::mutex> cache_lock(metrics_cache_mutex_);
    metrics_scratches_.resize(static_cast<std::size_t>(thread_count));
    metrics_predictions_.resize(static_cast<std::size_t>(thread_count));
    for (auto& predicted : metrics_predictions_) {
      predicted.resize(static_cast<std::size_t>(contour_count));
      for (auto& polygon : predicted) {
        polygon.resize(static_cast<std::size_t>(anchors_per_contour));
      }
    }
    {
      py::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(thread_count)
#endif
      for (py::ssize_t case_index = 0; case_index < case_count; ++case_index) {
#ifdef _OPENMP
        const int thread_index = omp_get_thread_num();
#else
        const int thread_index = 0;
#endif
        const double* values = input +
            static_cast<std::size_t>(case_index) * values_per_case;
        auto& predicted =
            metrics_predictions_[static_cast<std::size_t>(thread_index)];
        for (int contour = 0; contour < contour_count; ++contour) {
          auto& polygon = predicted[static_cast<std::size_t>(contour)];
          const int base = contour * anchors_per_contour * 2;
          for (int anchor = 0; anchor < anchors_per_contour; ++anchor) {
            const int offset = base + anchor * 2;
            polygon[static_cast<std::size_t>(anchor)] = cv::Point2d(
                values[offset],
                values[offset + 1]);
          }
        }
        const ExactMetricCounts metrics = cached_double_metric_counts(
            references_[static_cast<std::size_t>(frame_view(case_index))],
            predicted,
            &metrics_scratches_[static_cast<std::size_t>(thread_index)]);
        output_view(case_index, 0) = static_cast<double>(metrics.gt_area);
        output_view(case_index, 1) = static_cast<double>(metrics.pred_area);
        output_view(case_index, 2) = static_cast<double>(metrics.intersection);
        output_view(case_index, 3) = static_cast<double>(metrics.union_area);
        output_view(case_index, 4) = metrics.recall;
        output_view(case_index, 5) = metrics.precision;
        output_view(case_index, 6) = metrics.iou;
      }
    }
    return output;
  }

  py::array_t<double> catmull_metrics_batch(
      const py::array_t<
          std::int32_t,
          py::array::c_style | py::array::forcecast>& frame_indices,
      const py::array_t<
          double,
          py::array::c_style | py::array::forcecast>& controls,
      const py::array_t<
          double,
          py::array::c_style | py::array::forcecast>& sampling_matrix,
      const int requested_threads,
      const bool check_topology) const {
    if (frame_indices.ndim() != 1) {
      throw py::value_error("frame_indices must have shape (N,)");
    }
    if (controls.ndim() != 3 || controls.shape(2) != 2 ||
        controls.shape(0) != frame_indices.shape(0)) {
      throw py::value_error("controls must have shape (N, control_points, 2)");
    }
    if (sampling_matrix.ndim() != 2 ||
        sampling_matrix.shape(1) != controls.shape(1)) {
      throw py::value_error(
          "sampling_matrix must have shape (boundary_points, control_points)");
    }
    const py::ssize_t case_count = frame_indices.shape(0);
    const int control_count = static_cast<int>(controls.shape(1));
    const int boundary_count = static_cast<int>(sampling_matrix.shape(0));
    if (control_count < 3 || boundary_count < 3) {
      throw py::value_error("Catmull-Rom batches require at least three points");
    }
    const auto frame_view = frame_indices.unchecked<1>();
    for (py::ssize_t index = 0; index < case_count; ++index) {
      if (frame_view(index) < 0 ||
          frame_view(index) >= static_cast<int>(references_.size())) {
        throw py::value_error("frame index is outside the evaluator sequence");
      }
    }
    py::array_t<double> output({case_count, static_cast<py::ssize_t>(8)});
    auto output_view = output.mutable_unchecked<2>();
    const double* input = controls.data();
    const double* weights = sampling_matrix.data();
    const std::size_t control_values =
        static_cast<std::size_t>(control_count) * 2U;
    const int thread_count = std::max(1, requested_threads);
    std::lock_guard<std::mutex> cache_lock(catmull_cache_mutex_);
    catmull_scratches_.resize(static_cast<std::size_t>(thread_count));
    catmull_predictions_.resize(static_cast<std::size_t>(thread_count));
    catmull_topology_values_.resize(static_cast<std::size_t>(thread_count));
    for (int thread = 0; thread < thread_count; ++thread) {
      catmull_predictions_[static_cast<std::size_t>(thread)].resize(1);
      catmull_predictions_[static_cast<std::size_t>(thread)][0].resize(
          static_cast<std::size_t>(boundary_count));
      catmull_topology_values_[static_cast<std::size_t>(thread)].resize(
          static_cast<std::size_t>(boundary_count) * 2U);
    }
    {
      py::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 64) num_threads(thread_count)
#endif
      for (py::ssize_t case_index = 0; case_index < case_count; ++case_index) {
#ifdef _OPENMP
        const int thread_index = omp_get_thread_num();
#else
        const int thread_index = 0;
#endif
        const double* control = input +
            static_cast<std::size_t>(case_index) * control_values;
        auto& polygon =
            catmull_predictions_[static_cast<std::size_t>(thread_index)][0];
        auto& flat =
            catmull_topology_values_[static_cast<std::size_t>(thread_index)];
        for (int boundary = 0; boundary < boundary_count; ++boundary) {
          const double* row = weights +
              static_cast<std::size_t>(boundary) * control_count;
          double x = 0.0;
          double y = 0.0;
          for (int point = 0; point < control_count; ++point) {
            const double weight = row[point];
            x += weight * control[static_cast<std::size_t>(point) * 2U];
            y += weight * control[static_cast<std::size_t>(point) * 2U + 1U];
          }
          polygon[static_cast<std::size_t>(boundary)] = cv::Point2d(x, y);
          flat[static_cast<std::size_t>(boundary) * 2U] = x;
          flat[static_cast<std::size_t>(boundary) * 2U + 1U] = y;
        }
        const ExactMetricCounts metrics = cached_double_metric_counts(
            references_[static_cast<std::size_t>(frame_view(case_index))],
            catmull_predictions_[static_cast<std::size_t>(thread_index)],
            &catmull_scratches_[static_cast<std::size_t>(thread_index)]);
        output_view(case_index, 0) = static_cast<double>(metrics.gt_area);
        output_view(case_index, 1) = static_cast<double>(metrics.pred_area);
        output_view(case_index, 2) = static_cast<double>(metrics.intersection);
        output_view(case_index, 3) = static_cast<double>(metrics.union_area);
        output_view(case_index, 4) = metrics.recall;
        output_view(case_index, 5) = metrics.precision;
        output_view(case_index, 6) = metrics.iou;
        output_view(case_index, 7) =
            check_topology && has_strict_self_intersection_impl(
                flat.data(), static_cast<std::size_t>(boundary_count))
            ? 0.0
            : 1.0;
      }
    }
    return output;
  }

  py::array_t<std::uint8_t> catmull_topology_batch(
      const py::array_t<
          double,
          py::array::c_style | py::array::forcecast>& controls,
      const py::array_t<
          double,
          py::array::c_style | py::array::forcecast>& sampling_matrix,
      const int requested_threads) const {
    if (controls.ndim() != 3 || controls.shape(2) != 2) {
      throw py::value_error("controls must have shape (N, control_points, 2)");
    }
    if (sampling_matrix.ndim() != 2 ||
        sampling_matrix.shape(1) != controls.shape(1)) {
      throw py::value_error(
          "sampling_matrix must have shape (boundary_points, control_points)");
    }
    const py::ssize_t case_count = controls.shape(0);
    const int control_count = static_cast<int>(controls.shape(1));
    const int boundary_count = static_cast<int>(sampling_matrix.shape(0));
    py::array_t<std::uint8_t> output(case_count);
    std::uint8_t* output_values = output.mutable_data();
    const double* input = controls.data();
    const double* weights = sampling_matrix.data();
    const std::size_t control_values =
        static_cast<std::size_t>(control_count) * 2U;
    const int thread_count = std::max(1, requested_threads);
    std::lock_guard<std::mutex> cache_lock(catmull_cache_mutex_);
    catmull_topology_values_.resize(static_cast<std::size_t>(thread_count));
    for (int thread = 0; thread < thread_count; ++thread) {
      catmull_topology_values_[static_cast<std::size_t>(thread)].resize(
          static_cast<std::size_t>(boundary_count) * 2U);
    }
    {
      py::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 64) num_threads(thread_count)
#endif
      for (py::ssize_t case_index = 0; case_index < case_count; ++case_index) {
#ifdef _OPENMP
        const int thread_index = omp_get_thread_num();
#else
        const int thread_index = 0;
#endif
        const double* control = input +
            static_cast<std::size_t>(case_index) * control_values;
        auto& flat =
            catmull_topology_values_[static_cast<std::size_t>(thread_index)];
        for (int boundary = 0; boundary < boundary_count; ++boundary) {
          const double* row = weights +
              static_cast<std::size_t>(boundary) * control_count;
          double x = 0.0;
          double y = 0.0;
          for (int point = 0; point < control_count; ++point) {
            const double weight = row[point];
            x += weight * control[static_cast<std::size_t>(point) * 2U];
            y += weight * control[static_cast<std::size_t>(point) * 2U + 1U];
          }
          flat[static_cast<std::size_t>(boundary) * 2U] = x;
          flat[static_cast<std::size_t>(boundary) * 2U + 1U] = y;
        }
        output_values[case_index] = has_strict_self_intersection_impl(
            flat.data(), static_cast<std::size_t>(boundary_count))
            ? 0U
            : 1U;
      }
    }
    return output;
  }

  py::array_t<double> catmull_scale_metrics_batch(
      const py::array_t<
          double,
          py::array::c_style | py::array::forcecast>& controls,
      const py::array_t<
          double,
          py::array::c_style | py::array::forcecast>& scales,
      const py::array_t<
          double,
          py::array::c_style | py::array::forcecast>& sampling_matrix,
      const int requested_threads,
      const bool check_topology) const {
    if (controls.ndim() != 3 || controls.shape(2) != 2) {
      throw py::value_error("controls must have shape (frames, control_points, 2)");
    }
    if (scales.ndim() != 1 || scales.shape(0) <= 0) {
      throw py::value_error("scales must have shape (states,)");
    }
    if (sampling_matrix.ndim() != 2 ||
        sampling_matrix.shape(1) != controls.shape(1)) {
      throw py::value_error(
          "sampling_matrix must have shape (boundary_points, control_points)");
    }
    const int frame_count = static_cast<int>(controls.shape(0));
    const int control_count = static_cast<int>(controls.shape(1));
    const int state_count = static_cast<int>(scales.shape(0));
    const int boundary_count = static_cast<int>(sampling_matrix.shape(0));
    if (frame_count != static_cast<int>(references_.size()) ||
        control_count < 3 || boundary_count < 3) {
      throw py::value_error("Catmull-Rom scale dimensions do not match references");
    }
    py::array_t<double> output({
        static_cast<py::ssize_t>(frame_count),
        static_cast<py::ssize_t>(state_count),
        static_cast<py::ssize_t>(8)});
    auto output_view = output.mutable_unchecked<3>();
    const double* input = controls.data();
    const double* scale_values = scales.data();
    const double* weights = sampling_matrix.data();
    const std::size_t control_values =
        static_cast<std::size_t>(control_count) * 2U;
    const int case_count = frame_count * state_count;
    const int thread_count = std::max(1, requested_threads);
    std::vector<double> centers(static_cast<std::size_t>(frame_count) * 2U, 0.0);
    for (int frame = 0; frame < frame_count; ++frame) {
      const double* control = input +
          static_cast<std::size_t>(frame) * control_values;
      for (int point = 0; point < control_count; ++point) {
        centers[static_cast<std::size_t>(frame) * 2U] +=
            control[static_cast<std::size_t>(point) * 2U];
        centers[static_cast<std::size_t>(frame) * 2U + 1U] +=
            control[static_cast<std::size_t>(point) * 2U + 1U];
      }
      centers[static_cast<std::size_t>(frame) * 2U] /=
          static_cast<double>(control_count);
      centers[static_cast<std::size_t>(frame) * 2U + 1U] /=
          static_cast<double>(control_count);
    }
    std::lock_guard<std::mutex> cache_lock(catmull_cache_mutex_);
    catmull_scratches_.resize(static_cast<std::size_t>(thread_count));
    catmull_predictions_.resize(static_cast<std::size_t>(thread_count));
    catmull_topology_values_.resize(static_cast<std::size_t>(thread_count));
    catmull_scaled_controls_.resize(static_cast<std::size_t>(thread_count));
    for (int thread = 0; thread < thread_count; ++thread) {
      catmull_predictions_[static_cast<std::size_t>(thread)].resize(1);
      catmull_predictions_[static_cast<std::size_t>(thread)][0].resize(
          static_cast<std::size_t>(boundary_count));
      catmull_topology_values_[static_cast<std::size_t>(thread)].resize(
          static_cast<std::size_t>(boundary_count) * 2U);
      catmull_scaled_controls_[static_cast<std::size_t>(thread)].resize(
          control_values);
    }
    {
      py::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 64) num_threads(thread_count)
#endif
      for (int case_index = 0; case_index < case_count; ++case_index) {
#ifdef _OPENMP
        const int thread_index = omp_get_thread_num();
#else
        const int thread_index = 0;
#endif
        const int frame = case_index / state_count;
        const int state = case_index % state_count;
        const double* control = input +
            static_cast<std::size_t>(frame) * control_values;
        const double center_x = centers[static_cast<std::size_t>(frame) * 2U];
        const double center_y = centers[static_cast<std::size_t>(frame) * 2U + 1U];
        const double scale = scale_values[state];
        auto& scaled =
            catmull_scaled_controls_[static_cast<std::size_t>(thread_index)];
        for (int point = 0; point < control_count; ++point) {
          scaled[static_cast<std::size_t>(point) * 2U] = center_x + scale *
              (control[static_cast<std::size_t>(point) * 2U] - center_x);
          scaled[static_cast<std::size_t>(point) * 2U + 1U] = center_y + scale *
              (control[static_cast<std::size_t>(point) * 2U + 1U] - center_y);
        }
        auto& polygon =
            catmull_predictions_[static_cast<std::size_t>(thread_index)][0];
        auto& flat =
            catmull_topology_values_[static_cast<std::size_t>(thread_index)];
        for (int boundary = 0; boundary < boundary_count; ++boundary) {
          const double* row = weights +
              static_cast<std::size_t>(boundary) * control_count;
          double x = 0.0;
          double y = 0.0;
          for (int point = 0; point < control_count; ++point) {
            x += row[point] * scaled[static_cast<std::size_t>(point) * 2U];
            y += row[point] * scaled[static_cast<std::size_t>(point) * 2U + 1U];
          }
          polygon[static_cast<std::size_t>(boundary)] = cv::Point2d(x, y);
          flat[static_cast<std::size_t>(boundary) * 2U] = x;
          flat[static_cast<std::size_t>(boundary) * 2U + 1U] = y;
        }
        const ExactMetricCounts metrics = cached_double_metric_counts(
            references_[static_cast<std::size_t>(frame)],
            catmull_predictions_[static_cast<std::size_t>(thread_index)],
            &catmull_scratches_[static_cast<std::size_t>(thread_index)]);
        output_view(frame, state, 0) = static_cast<double>(metrics.gt_area);
        output_view(frame, state, 1) = static_cast<double>(metrics.pred_area);
        output_view(frame, state, 2) = static_cast<double>(metrics.intersection);
        output_view(frame, state, 3) = static_cast<double>(metrics.union_area);
        output_view(frame, state, 4) = metrics.recall;
        output_view(frame, state, 5) = metrics.precision;
        output_view(frame, state, 6) = metrics.iou;
        output_view(frame, state, 7) =
            check_topology && has_strict_self_intersection_impl(
                flat.data(), static_cast<std::size_t>(boundary_count))
            ? 0.0
            : 1.0;
      }
    }
    return output;
  }

  py::array_t<double> edge_metrics_batch(
      const py::array_t<
          double,
          py::array::c_style | py::array::forcecast>& candidate_boundaries,
      const py::array_t<
          std::int32_t,
          py::array::c_style | py::array::forcecast>& edges,
      const double recall_floor,
      const double low_iou_quadratic_weight,
      const int requested_threads,
      const bool check_topology) const {
    if (candidate_boundaries.ndim() != 4 ||
        candidate_boundaries.shape(3) != 2) {
      throw py::value_error(
          "candidate_boundaries must have shape (frames, states, points, 2)");
    }
    if (edges.ndim() != 2 || edges.shape(1) != 4) {
      throw py::value_error("edges must have shape (E, 4)");
    }
    const int frame_count = static_cast<int>(candidate_boundaries.shape(0));
    const int state_count = static_cast<int>(candidate_boundaries.shape(1));
    const int point_count = static_cast<int>(candidate_boundaries.shape(2));
    if (frame_count != static_cast<int>(references_.size()) || point_count < 3) {
      throw py::value_error("candidate boundary dimensions do not match references");
    }
    const auto edge_view = edges.unchecked<2>();
    const py::ssize_t edge_count = edges.shape(0);
    for (py::ssize_t index = 0; index < edge_count; ++index) {
      const int start_frame = edge_view(index, 0);
      const int start_state = edge_view(index, 1);
      const int end_frame = edge_view(index, 2);
      const int end_state = edge_view(index, 3);
      if (start_frame < 0 || start_frame >= end_frame || end_frame >= frame_count ||
          start_state < 0 || start_state >= state_count ||
          end_state < 0 || end_state >= state_count) {
        throw py::value_error("edge index is outside candidate boundary dimensions");
      }
    }
    py::array_t<double> output({edge_count, static_cast<py::ssize_t>(5)});
    auto output_view = output.mutable_unchecked<2>();
    const double* values = candidate_boundaries.data();
    const std::size_t values_per_state =
        static_cast<std::size_t>(point_count) * 2U;
    const std::size_t values_per_frame =
        static_cast<std::size_t>(state_count) * values_per_state;
    const int thread_count = std::max(1, requested_threads);
    std::vector<DoubleRasterScratch> scratches(
        static_cast<std::size_t>(thread_count));
    // Every graph edge ending at the same (frame, state) has the exact same
    // final-frame raster.  The former loop rasterized that endpoint once per
    // possible predecessor, which dominates wide DP graphs.  Precompute each
    // endpoint exactly once.  Loss accumulation still appends this value in
    // the same final position, preserving floating-point addition order.
    std::vector<ExactMetricCounts> endpoint_metrics(
        static_cast<std::size_t>(frame_count) * state_count);
    std::vector<std::uint8_t> endpoint_topology(
        static_cast<std::size_t>(frame_count) * state_count,
        static_cast<std::uint8_t>(1));
    std::vector<std::vector<DoublePolygon>> endpoint_predictions(
        static_cast<std::size_t>(thread_count),
        std::vector<DoublePolygon>(1));
    for (auto& prediction : endpoint_predictions) {
      prediction[0].resize(static_cast<std::size_t>(point_count));
    }
    const int endpoint_count = std::max(0, frame_count - 1) * state_count;
    {
      py::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 64) num_threads(thread_count)
#endif
      for (int endpoint_index = 0; endpoint_index < endpoint_count;
           ++endpoint_index) {
#ifdef _OPENMP
        const int thread_index = omp_get_thread_num();
#else
        const int thread_index = 0;
#endif
        const int frame = 1 + endpoint_index / state_count;
        const int state = endpoint_index % state_count;
        const double* endpoint = values +
            static_cast<std::size_t>(frame) * values_per_frame +
            static_cast<std::size_t>(state) * values_per_state;
        auto& predicted =
            endpoint_predictions[static_cast<std::size_t>(thread_index)];
        for (int point = 0; point < point_count; ++point) {
          const std::size_t offset = static_cast<std::size_t>(point) * 2U;
          predicted[0][static_cast<std::size_t>(point)] = cv::Point2d(
              endpoint[offset], endpoint[offset + 1U]);
        }
        const std::size_t output_index =
            static_cast<std::size_t>(frame) * state_count + state;
        endpoint_metrics[output_index] = cached_double_metric_counts(
            references_[static_cast<std::size_t>(frame)],
            predicted,
            &scratches[static_cast<std::size_t>(thread_index)]);
        if (check_topology) {
          endpoint_topology[output_index] =
              has_strict_self_intersection_impl(
                  endpoint, static_cast<std::size_t>(point_count))
              ? static_cast<std::uint8_t>(0)
              : static_cast<std::uint8_t>(1);
        }
      }
    }
    // Reuse endpoint interpolation storage per worker instead of allocating a
    // 288-point vector for every edge.  Each OpenMP worker exclusively owns
    // its slot.
    std::vector<std::vector<DoublePolygon>> predictions(
        static_cast<std::size_t>(thread_count),
        std::vector<DoublePolygon>(1));
    for (auto& prediction : predictions) {
      prediction[0].resize(static_cast<std::size_t>(point_count));
    }
    std::vector<std::vector<double>> topology_values(
        static_cast<std::size_t>(thread_count));
    if (check_topology) {
      for (auto& flat : topology_values) {
        flat.resize(values_per_state);
      }
    }
    {
      py::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 32) num_threads(thread_count)
#endif
      for (py::ssize_t edge_index = 0; edge_index < edge_count; ++edge_index) {
#ifdef _OPENMP
        const int thread_index = omp_get_thread_num();
#else
        const int thread_index = 0;
#endif
        const int start_frame = edge_view(edge_index, 0);
        const int start_state = edge_view(edge_index, 1);
        const int end_frame = edge_view(edge_index, 2);
        const int end_state = edge_view(edge_index, 3);
        const double* start = values +
            static_cast<std::size_t>(start_frame) * values_per_frame +
            static_cast<std::size_t>(start_state) * values_per_state;
        const double* end = values +
            static_cast<std::size_t>(end_frame) * values_per_frame +
            static_cast<std::size_t>(end_state) * values_per_state;
        double loss_total = 0.0;
        double iou_total = 0.0;
        double minimum_recall = 1.0;
        int frames_covered = 0;
        bool topology_valid = true;
        bool continue_to_endpoint = true;
        auto& predicted = predictions[static_cast<std::size_t>(thread_index)];
        auto& flat = topology_values[static_cast<std::size_t>(thread_index)];
        for (int frame = start_frame + 1; frame < end_frame; ++frame) {
          const double alpha =
              static_cast<double>(frame - start_frame) /
              static_cast<double>(end_frame - start_frame);
          const double beta = 1.0 - alpha;
          for (int point = 0; point < point_count; ++point) {
            const std::size_t offset = static_cast<std::size_t>(point) * 2U;
            const double x = beta * start[offset] + alpha * end[offset];
            const double y = beta * start[offset + 1U] + alpha * end[offset + 1U];
            predicted[0][static_cast<std::size_t>(point)] = cv::Point2d(x, y);
            if (check_topology) {
              flat[offset] = x;
              flat[offset + 1U] = y;
            }
          }
          if (check_topology && has_strict_self_intersection_impl(
                  flat.data(), static_cast<std::size_t>(point_count))) {
            topology_valid = false;
            continue_to_endpoint = false;
            break;
          }
          const ExactMetricCounts metrics = cached_double_metric_counts(
              references_[static_cast<std::size_t>(frame)],
              predicted,
              &scratches[static_cast<std::size_t>(thread_index)]);
          ++frames_covered;
          minimum_recall = std::min(minimum_recall, metrics.recall);
          if (metrics.recall + 1e-12 < recall_floor) {
            continue_to_endpoint = false;
            break;
          }
          const double loss = 1.0 - metrics.iou;
          loss_total += loss + low_iou_quadratic_weight * loss * loss;
          iou_total += metrics.iou;
        }
        if (continue_to_endpoint) {
          const std::size_t endpoint_index =
              static_cast<std::size_t>(end_frame) * state_count + end_state;
          if (check_topology && !endpoint_topology[endpoint_index]) {
            topology_valid = false;
          } else {
            const ExactMetricCounts& metrics = endpoint_metrics[endpoint_index];
            ++frames_covered;
            minimum_recall = std::min(minimum_recall, metrics.recall);
            if (metrics.recall + 1e-12 >= recall_floor) {
              const double loss = 1.0 - metrics.iou;
              loss_total += loss + low_iou_quadratic_weight * loss * loss;
              iou_total += metrics.iou;
            }
          }
        }
        output_view(edge_index, 0) = loss_total;
        output_view(edge_index, 1) = iou_total;
        output_view(edge_index, 2) = minimum_recall;
        output_view(edge_index, 3) = static_cast<double>(frames_covered);
        output_view(edge_index, 4) = topology_valid ? 1.0 : 0.0;
      }
    }
    return output;
  }

 private:
  std::vector<DoubleFrameReference> references_;
  mutable std::mutex metrics_cache_mutex_;
  mutable std::vector<DoubleRasterScratch> metrics_scratches_;
  mutable std::vector<std::vector<DoublePolygon>> metrics_predictions_;
  mutable std::mutex catmull_cache_mutex_;
  mutable std::vector<DoubleRasterScratch> catmull_scratches_;
  mutable std::vector<std::vector<DoublePolygon>> catmull_predictions_;
  mutable std::vector<std::vector<double>> catmull_topology_values_;
  mutable std::vector<std::vector<double>> catmull_scaled_controls_;
  std::size_t cached_reference_frames_ = 0U;
  std::size_t run_cached_reference_frames_ = 0U;
  std::size_t bitmap_cached_reference_frames_ = 0U;
  std::uint64_t cached_reference_bytes_ = 0U;
  std::uint64_t total_reference_cache_bytes_ = 0U;
  std::uint64_t total_bitmap_reference_bytes_ = 0U;
  std::uint64_t cache_budget_bytes_ = 0U;
};

double shape_distance_impl(
    const float* src,
    const float* dst,
    const std::size_t points,
    const double normalization_scale) {
  if (points == 0) {
    return 0.0;
  }
  double src_mean_x = 0.0;
  double src_mean_y = 0.0;
  double dst_mean_x = 0.0;
  double dst_mean_y = 0.0;
  for (std::size_t index = 0; index < points; ++index) {
    src_mean_x += static_cast<double>(src[index * 2]);
    src_mean_y += static_cast<double>(src[index * 2 + 1]);
    dst_mean_x += static_cast<double>(dst[index * 2]);
    dst_mean_y += static_cast<double>(dst[index * 2 + 1]);
  }
  const double inverse_count = 1.0 / static_cast<double>(points);
  src_mean_x *= inverse_count;
  src_mean_y *= inverse_count;
  dst_mean_x *= inverse_count;
  dst_mean_y *= inverse_count;

  double source_variance_sum = 0.0;
  double covariance_00_sum = 0.0;
  double covariance_01_sum = 0.0;
  double covariance_10_sum = 0.0;
  double covariance_11_sum = 0.0;
  for (std::size_t index = 0; index < points; ++index) {
    const double sx = static_cast<double>(src[index * 2]) - src_mean_x;
    const double sy = static_cast<double>(src[index * 2 + 1]) - src_mean_y;
    const double dx = static_cast<double>(dst[index * 2]) - dst_mean_x;
    const double dy = static_cast<double>(dst[index * 2 + 1]) - dst_mean_y;
    source_variance_sum += sx * sx + sy * sy;
    covariance_00_sum += dx * sx;
    covariance_01_sum += dx * sy;
    covariance_10_sum += dy * sx;
    covariance_11_sum += dy * sy;
  }
  const double source_variance = source_variance_sum * inverse_count;
  double similarity_scale = 1.0;
  double cosine = 1.0;
  double sine = 0.0;
  if (points >= 2 && source_variance > 1e-9) {
    const double covariance_trace =
        (covariance_00_sum + covariance_11_sum) * inverse_count;
    const double covariance_skew =
        (covariance_10_sum - covariance_01_sum) * inverse_count;
    const double proper_singular_sum = std::hypot(covariance_trace, covariance_skew);
    if (proper_singular_sum > 0.0) {
      cosine = covariance_trace / proper_singular_sum;
      sine = covariance_skew / proper_singular_sum;
      similarity_scale = proper_singular_sum / std::max(source_variance, 1e-9);
      if (!std::isfinite(similarity_scale) || similarity_scale <= 1e-9) {
        similarity_scale = 1.0;
      }
    }
  }
  const double translation_x =
      dst_mean_x - similarity_scale * (cosine * src_mean_x - sine * src_mean_y);
  const double translation_y =
      dst_mean_y - similarity_scale * (sine * src_mean_x + cosine * src_mean_y);

  double norm_sum = 0.0;
  for (std::size_t index = 0; index < points; ++index) {
    const double sx = static_cast<double>(src[index * 2]);
    const double sy = static_cast<double>(src[index * 2 + 1]);
    // Production casts the transformed coordinates and then the residuals to
    // float32 before computing the final float64 norm.
    const float aligned_x = static_cast<float>(
        similarity_scale * (cosine * sx - sine * sy) + translation_x);
    const float aligned_y = static_cast<float>(
        similarity_scale * (sine * sx + cosine * sy) + translation_y);
    const float residual_x = static_cast<float>(
        static_cast<double>(dst[index * 2]) - static_cast<double>(aligned_x));
    const float residual_y = static_cast<float>(
        static_cast<double>(dst[index * 2 + 1]) - static_cast<double>(aligned_y));
    norm_sum += std::hypot(
        static_cast<double>(residual_x),
        static_cast<double>(residual_y));
  }
  return (norm_sum * inverse_count) / std::max(normalization_scale, 1.0);
}

double shape_distance(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& source,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& destination,
    const double normalization_scale) {
  if (source.size() != destination.size() || source.size() % 2 != 0) {
    throw py::value_error("shape vectors must have equal (N, 2) sizes");
  }
  return shape_distance_impl(
      source.data(),
      destination.data(),
      static_cast<std::size_t>(source.size() / 2),
      normalization_scale);
}

py::tuple decode_penalty_path(
    const py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>& edges,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& edge_costs,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& initial_losses,
    const int frame_count,
    const int state_count,
    const double penalty) {
  if (edges.ndim() != 2 || edges.shape(1) != 4) {
    throw py::value_error("edges must have shape (E, 4)");
  }
  if (edge_costs.ndim() != 1 || edge_costs.shape(0) != edges.shape(0)) {
    throw py::value_error("edge_costs must have shape (E,)");
  }
  if (initial_losses.ndim() != 1 || initial_losses.shape(0) != state_count) {
    throw py::value_error("initial_losses must have shape (states,)");
  }
  if (frame_count <= 0 || state_count <= 0) {
    throw py::value_error("frame_count and state_count must be positive");
  }
  const auto edge_view = edges.unchecked<2>();
  const auto edge_cost_view = edge_costs.unchecked<1>();
  const auto initial_view = initial_losses.unchecked<1>();
  const std::size_t node_values =
      static_cast<std::size_t>(frame_count) * static_cast<std::size_t>(state_count);
  const double infinity = std::numeric_limits<double>::infinity();
  std::vector<double> costs(node_values, infinity);
  std::vector<double> raw_costs(node_values, infinity);
  std::vector<std::int32_t> counts(node_values, 1 << 30);
  std::vector<std::int32_t> back_pos(node_values, -1);
  std::vector<std::int16_t> back_state(node_values, -1);
  const auto index_of = [state_count](const int frame, const int state) {
    return static_cast<std::size_t>(frame) * static_cast<std::size_t>(state_count)
        + static_cast<std::size_t>(state);
  };
  for (int state = 0; state < state_count; ++state) {
    const double raw = initial_view(state);
    if (!std::isfinite(raw)) {
      continue;
    }
    const std::size_t index = index_of(0, state);
    costs[index] = raw + penalty;
    raw_costs[index] = raw;
    counts[index] = 1;
  }
  {
    py::gil_scoped_release release;
    for (py::ssize_t edge_index = 0; edge_index < edges.shape(0); ++edge_index) {
      const double edge_cost = edge_cost_view(edge_index);
      if (!std::isfinite(edge_cost)) {
        continue;
      }
      const int start_frame = edge_view(edge_index, 0);
      const int start_state = edge_view(edge_index, 1);
      const int end_frame = edge_view(edge_index, 2);
      const int end_state = edge_view(edge_index, 3);
      if (start_frame < 0 || start_frame >= end_frame || end_frame >= frame_count ||
          start_state < 0 || start_state >= state_count ||
          end_state < 0 || end_state >= state_count) {
        continue;
      }
      const std::size_t start_index = index_of(start_frame, start_state);
      if (!std::isfinite(costs[start_index])) {
        continue;
      }
      const double candidate_raw = raw_costs[start_index] + edge_cost;
      const std::int32_t candidate_count = counts[start_index] + 1;
      const double candidate_cost = candidate_raw + penalty * candidate_count;
      const std::size_t end_index = index_of(end_frame, end_state);
      const double current_cost = costs[end_index];
      const double current_raw = raw_costs[end_index];
      const std::int32_t current_count = counts[end_index];
      if (candidate_cost < current_cost - 1e-12 ||
          (std::abs(candidate_cost - current_cost) <= 1e-12 &&
           (candidate_raw < current_raw - 1e-12 ||
            (std::abs(candidate_raw - current_raw) <= 1e-12 &&
             candidate_count < current_count)))) {
        costs[end_index] = candidate_cost;
        raw_costs[end_index] = candidate_raw;
        counts[end_index] = candidate_count;
        back_pos[end_index] = start_frame;
        back_state[end_index] = static_cast<std::int16_t>(start_state);
      }
    }
  }
  int final_state = -1;
  for (int state = 0; state < state_count; ++state) {
    const std::size_t index = index_of(frame_count - 1, state);
    if (!std::isfinite(costs[index])) {
      continue;
    }
    if (final_state < 0) {
      final_state = state;
      continue;
    }
    const std::size_t best = index_of(frame_count - 1, final_state);
    if (std::tie(costs[index], raw_costs[index], counts[index], state) <
        std::tie(costs[best], raw_costs[best], counts[best], final_state)) {
      final_state = state;
    }
  }
  if (final_state < 0) {
    return py::make_tuple(py::list(), py::list(), infinity);
  }
  std::vector<int> positions;
  std::vector<int> states;
  int position = frame_count - 1;
  int state = final_state;
  while (position >= 0) {
    positions.push_back(position);
    states.push_back(state);
    if (position == 0) {
      break;
    }
    const std::size_t index = index_of(position, state);
    position = back_pos[index];
    state = static_cast<int>(back_state[index]);
    if (position < 0 || state < 0) {
      throw std::runtime_error("broken native DP predecessor chain");
    }
  }
  std::reverse(positions.begin(), positions.end());
  std::reverse(states.begin(), states.end());
  return py::make_tuple(positions, states, raw_costs[index_of(frame_count - 1, final_state)]);
}

class IncrementalPenaltyPathDecoder {
 public:
  IncrementalPenaltyPathDecoder(
      const py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>& edges,
      const py::array_t<double, py::array::c_style | py::array::forcecast>& initial_losses,
      const int frame_count,
      const int state_count)
      : frame_count_(frame_count), state_count_(state_count) {
    if (edges.ndim() != 2 || edges.shape(1) != 4) {
      throw py::value_error("edges must have shape (E, 4)");
    }
    if (initial_losses.ndim() != 1 || initial_losses.shape(0) != state_count) {
      throw py::value_error("initial_losses must have shape (states,)");
    }
    if (frame_count <= 0 || state_count <= 0) {
      throw py::value_error("frame_count and state_count must be positive");
    }
    const auto edge_view = edges.unchecked<2>();
    edge_count_ = static_cast<std::size_t>(edges.shape(0));
    edges_.resize(edge_count_ * 4);
    int previous_end = -1;
    for (std::size_t index = 0; index < edge_count_; ++index) {
      for (int column = 0; column < 4; ++column) {
        edges_[index * 4 + static_cast<std::size_t>(column)] =
            edge_view(static_cast<py::ssize_t>(index), column);
      }
      const int start_frame = edges_[index * 4];
      const int start_state = edges_[index * 4 + 1];
      const int end_frame = edges_[index * 4 + 2];
      const int end_state = edges_[index * 4 + 3];
      if (start_frame < 0 || start_frame >= end_frame || end_frame >= frame_count_ ||
          start_state < 0 || start_state >= state_count_ ||
          end_state < 0 || end_state >= state_count_) {
        throw py::value_error("edge is outside the decoder graph");
      }
      if (end_frame < previous_end) {
        throw py::value_error("edges must be ordered by non-decreasing end frame");
      }
      previous_end = end_frame;
    }
    const auto initial_view = initial_losses.unchecked<1>();
    initial_losses_.resize(static_cast<std::size_t>(state_count_));
    for (int state = 0; state < state_count_; ++state) {
      initial_losses_[static_cast<std::size_t>(state)] = initial_view(state);
    }
    const std::size_t node_values =
        static_cast<std::size_t>(frame_count_) * static_cast<std::size_t>(state_count_);
    costs_.resize(node_values);
    raw_costs_.resize(node_values);
    counts_.resize(node_values);
    back_pos_.resize(node_values);
    back_state_.resize(node_values);
    first_edge_by_end_.assign(static_cast<std::size_t>(frame_count_ + 1), edge_count_);
    std::size_t cursor = 0;
    for (int frame = 0; frame <= frame_count_; ++frame) {
      while (cursor < edge_count_ && edges_[cursor * 4 + 2] < frame) {
        ++cursor;
      }
      first_edge_by_end_[static_cast<std::size_t>(frame)] = cursor;
    }
  }

  py::tuple decode(
      const py::array_t<double, py::array::c_style | py::array::forcecast>& edge_costs,
      const double penalty,
      const int requested_recompute_from) {
    if (edge_costs.ndim() != 1 ||
        static_cast<std::size_t>(edge_costs.shape(0)) != edge_count_) {
      throw py::value_error("edge_costs must have shape (E,)");
    }
    int recompute_from = std::clamp(requested_recompute_from, 0, frame_count_ - 1);
    if (!initialized_ || penalty != penalty_) {
      recompute_from = 0;
    }
    const auto edge_cost_view = edge_costs.unchecked<1>();
    const double infinity = std::numeric_limits<double>::infinity();
    const auto index_of = [this](const int frame, const int state) {
      return static_cast<std::size_t>(frame) * static_cast<std::size_t>(state_count_)
          + static_cast<std::size_t>(state);
    };
    {
      py::gil_scoped_release release;
      const std::size_t first_node =
          static_cast<std::size_t>(recompute_from) * static_cast<std::size_t>(state_count_);
      std::fill(costs_.begin() + static_cast<std::ptrdiff_t>(first_node), costs_.end(), infinity);
      std::fill(
          raw_costs_.begin() + static_cast<std::ptrdiff_t>(first_node),
          raw_costs_.end(),
          infinity);
      std::fill(
          counts_.begin() + static_cast<std::ptrdiff_t>(first_node),
          counts_.end(),
          1 << 30);
      std::fill(
          back_pos_.begin() + static_cast<std::ptrdiff_t>(first_node),
          back_pos_.end(),
          -1);
      std::fill(
          back_state_.begin() + static_cast<std::ptrdiff_t>(first_node),
          back_state_.end(),
          static_cast<std::int16_t>(-1));
      if (recompute_from == 0) {
        for (int state = 0; state < state_count_; ++state) {
          const double raw = initial_losses_[static_cast<std::size_t>(state)];
          if (!std::isfinite(raw)) {
            continue;
          }
          const std::size_t index = index_of(0, state);
          costs_[index] = raw + penalty;
          raw_costs_[index] = raw;
          counts_[index] = 1;
        }
      }
      const std::size_t first_edge =
          first_edge_by_end_[static_cast<std::size_t>(recompute_from)];
      for (std::size_t edge_index = first_edge; edge_index < edge_count_; ++edge_index) {
        const double edge_cost = edge_cost_view(static_cast<py::ssize_t>(edge_index));
        if (!std::isfinite(edge_cost)) {
          continue;
        }
        const int start_frame = edges_[edge_index * 4];
        const int start_state = edges_[edge_index * 4 + 1];
        const int end_frame = edges_[edge_index * 4 + 2];
        const int end_state = edges_[edge_index * 4 + 3];
        const std::size_t start_index = index_of(start_frame, start_state);
        if (!std::isfinite(costs_[start_index])) {
          continue;
        }
        const double candidate_raw = raw_costs_[start_index] + edge_cost;
        const std::int32_t candidate_count = counts_[start_index] + 1;
        const double candidate_cost = candidate_raw + penalty * candidate_count;
        const std::size_t end_index = index_of(end_frame, end_state);
        const double current_cost = costs_[end_index];
        const double current_raw = raw_costs_[end_index];
        const std::int32_t current_count = counts_[end_index];
        if (candidate_cost < current_cost - 1e-12 ||
            (std::abs(candidate_cost - current_cost) <= 1e-12 &&
             (candidate_raw < current_raw - 1e-12 ||
              (std::abs(candidate_raw - current_raw) <= 1e-12 &&
               candidate_count < current_count)))) {
          costs_[end_index] = candidate_cost;
          raw_costs_[end_index] = candidate_raw;
          counts_[end_index] = candidate_count;
          back_pos_[end_index] = start_frame;
          back_state_[end_index] = static_cast<std::int16_t>(start_state);
        }
      }
    }
    initialized_ = true;
    penalty_ = penalty;
    return current_result();
  }

 private:
  py::tuple current_result() const {
    const double infinity = std::numeric_limits<double>::infinity();
    const auto index_of = [this](const int frame, const int state) {
      return static_cast<std::size_t>(frame) * static_cast<std::size_t>(state_count_)
          + static_cast<std::size_t>(state);
    };
    int final_state = -1;
    for (int state = 0; state < state_count_; ++state) {
      const std::size_t index = index_of(frame_count_ - 1, state);
      if (!std::isfinite(costs_[index])) {
        continue;
      }
      if (final_state < 0) {
        final_state = state;
        continue;
      }
      const std::size_t best = index_of(frame_count_ - 1, final_state);
      if (std::tie(costs_[index], raw_costs_[index], counts_[index], state) <
          std::tie(costs_[best], raw_costs_[best], counts_[best], final_state)) {
        final_state = state;
      }
    }
    if (final_state < 0) {
      return py::make_tuple(py::list(), py::list(), infinity);
    }
    std::vector<int> positions;
    std::vector<int> states;
    int position = frame_count_ - 1;
    int state = final_state;
    while (position >= 0) {
      positions.push_back(position);
      states.push_back(state);
      if (position == 0) {
        break;
      }
      const std::size_t index = index_of(position, state);
      position = back_pos_[index];
      state = static_cast<int>(back_state_[index]);
      if (position < 0 || state < 0) {
        throw std::runtime_error("broken incremental DP predecessor chain");
      }
    }
    std::reverse(positions.begin(), positions.end());
    std::reverse(states.begin(), states.end());
    return py::make_tuple(
        positions,
        states,
        raw_costs_[index_of(frame_count_ - 1, final_state)]);
  }
  int frame_count_ = 0;
  int state_count_ = 0;
  std::size_t edge_count_ = 0;
  std::vector<std::int32_t> edges_;
  std::vector<double> initial_losses_;
  std::vector<std::size_t> first_edge_by_end_;
  std::vector<double> costs_;
  std::vector<double> raw_costs_;
  std::vector<std::int32_t> counts_;
  std::vector<std::int32_t> back_pos_;
  std::vector<std::int16_t> back_state_;
  bool initialized_ = false;
  double penalty_ = 0.0;
};

class CachedIntervalEvaluator {
 public:
  CachedIntervalEvaluator(
      const py::iterable& gt_masks,
      const py::array_t<float, py::array::c_style | py::array::forcecast>& shifts,
      const py::array_t<float, py::array::c_style | py::array::forcecast>& scales,
      const py::iterable& exact_gt_frames) {
    if (shifts.ndim() != 2 || shifts.shape(1) != 2) {
      throw py::value_error("shifts must have shape (frames, 2)");
    }
    if (scales.ndim() != 1 || scales.shape(0) != shifts.shape(0)) {
      throw py::value_error("scales must have shape (frames,)");
    }
    const py::list masks(gt_masks);
    if (static_cast<py::ssize_t>(masks.size()) != shifts.shape(0)) {
      throw py::value_error("gt_masks, shifts and scales must have equal frame counts");
    }
    const py::list exact_frames(exact_gt_frames);
    if (exact_frames.size() != masks.size()) {
      throw py::value_error("exact_gt_frames must match the cached frame count");
    }

    const auto shift_view = shifts.unchecked<2>();
    const auto scale_view = scales.unchecked<1>();
    contexts_.reserve(masks.size());
    for (py::ssize_t index = 0; index < static_cast<py::ssize_t>(masks.size()); ++index) {
      const auto array = py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>::ensure(
          masks[index]);
      if (!array || array.ndim() != 2) {
        throw py::value_error("every gt_mask must be a C-contiguous uint8 2-D array");
      }
      const int height = static_cast<int>(array.shape(0));
      const int width = static_cast<int>(array.shape(1));
      cv::Mat view(
          height,
          width,
          CV_8UC1,
          const_cast<std::uint8_t*>(array.data()),
          static_cast<std::size_t>(array.strides(0)));
      CachedFrameContext context;
      context.gt_mask = view.clone();
      context.pred_mask = cv::Mat::zeros(height, width, CV_8UC1);
      context.intersection_mask = cv::Mat::zeros(height, width, CV_8UC1);
      context.gt_area = cv::countNonZero(context.gt_mask);
      context.shift_x = shift_view(index, 0);
      context.shift_y = shift_view(index, 1);
      context.scale_factor = scale_view(index);
      contexts_.push_back(std::move(context));
      auto exact_polygons = parse_polygons(
          py::reinterpret_borrow<py::iterable>(exact_frames[index]));
      exact_references_.push_back(build_exact_frame_reference(exact_polygons));
    }
  }

  py::tuple evaluate_vectors(
      const py::array_t<float, py::array::c_style | py::array::forcecast>& start_vector,
      const py::array_t<float, py::array::c_style | py::array::forcecast>& end_vector,
      const int contour_count,
      const int anchors_per_contour,
      const int start_index,
      const int end_index,
      const bool include_start,
      const double iou_weight,
      const double recall_floor) {
    if (contour_count < 0 || anchors_per_contour < 0) {
      throw py::value_error("contour counts must be non-negative");
    }
    if (start_index < 0 || end_index < start_index ||
        end_index >= static_cast<int>(contexts_.size())) {
      throw py::value_error("invalid interval bounds");
    }
    const py::ssize_t expected_values =
        static_cast<py::ssize_t>(contour_count) * anchors_per_contour * 2;
    if (start_vector.size() != expected_values || end_vector.size() != expected_values) {
      throw py::value_error("vector size does not match contour topology");
    }
    const CachedMetricsTotals totals = evaluate_cached_impl(
        start_vector.data(),
        end_vector.data(),
        contour_count,
        anchors_per_contour,
        start_index,
        end_index,
        include_start,
        iou_weight,
        recall_floor,
        nullptr,
        nullptr);
    return py::make_tuple(
        totals.frame_loss_total,
        totals.recall_deficit_total,
        totals.frames_covered);
  }

  py::tuple exact_frame_metrics(
      const int frame_index,
      const py::array_t<float, py::array::c_style | py::array::forcecast>& vector,
      const int contour_count,
      const int anchors_per_contour) {
    if (frame_index < 0 || frame_index >= static_cast<int>(exact_references_.size())) {
      throw py::value_error("frame_index is outside the evaluator sequence");
    }
    if (contour_count < 0 || anchors_per_contour < 0) {
      throw py::value_error("contour counts must be non-negative");
    }
    const py::ssize_t expected_values =
        static_cast<py::ssize_t>(contour_count) * anchors_per_contour * 2;
    if (vector.size() != expected_values) {
      throw py::value_error("vector size does not match contour topology");
    }
    const float* values = vector.data();
    std::vector<Polygon> predicted(static_cast<std::size_t>(contour_count));
    for (int contour = 0; contour < contour_count; ++contour) {
      auto& polygon = predicted[static_cast<std::size_t>(contour)];
      polygon.reserve(static_cast<std::size_t>(anchors_per_contour));
      const int base = contour * anchors_per_contour * 2;
      for (int anchor = 0; anchor < anchors_per_contour; ++anchor) {
        const int offset = base + anchor * 2;
        polygon.emplace_back(values[offset], values[offset + 1]);
      }
    }
    const ExactMetricCounts metrics = exact_metric_counts_from_reference(
        exact_references_[static_cast<std::size_t>(frame_index)],
        predicted,
        &endpoint_exact_scratch_);
    return py::make_tuple(
        static_cast<double>(metrics.gt_area),
        static_cast<double>(metrics.pred_area),
        static_cast<double>(metrics.intersection),
        static_cast<double>(metrics.union_area),
        metrics.recall,
        metrics.precision,
        metrics.iou);
  }

  py::array_t<double> exact_frame_metrics_batch(
      const py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>& frame_indices,
      const py::array_t<float, py::array::c_style | py::array::forcecast>& vectors,
      const int contour_count,
      const int anchors_per_contour,
      const int requested_threads) const {
    if (frame_indices.ndim() != 1) {
      throw py::value_error("frame_indices must have shape (N,)");
    }
    if (vectors.ndim() != 3 || vectors.shape(2) != 2) {
      throw py::value_error("vectors must have shape (N, points, 2)");
    }
    if (vectors.shape(0) != frame_indices.shape(0)) {
      throw py::value_error("frame_indices and vectors must have equal case counts");
    }
    if (contour_count < 0 || anchors_per_contour < 0) {
      throw py::value_error("contour counts must be non-negative");
    }
    const py::ssize_t expected_points =
        static_cast<py::ssize_t>(contour_count) * anchors_per_contour;
    if (vectors.shape(1) != expected_points) {
      throw py::value_error("vector point count does not match contour topology");
    }
    const py::ssize_t case_count = frame_indices.shape(0);
    const auto frame_view = frame_indices.unchecked<1>();
    for (py::ssize_t index = 0; index < case_count; ++index) {
      if (frame_view(index) < 0 ||
          frame_view(index) >= static_cast<int>(exact_references_.size())) {
        throw py::value_error("frame_index is outside the evaluator sequence");
      }
    }

    py::array_t<double> output({case_count, static_cast<py::ssize_t>(7)});
    auto output_view = output.mutable_unchecked<2>();
    const float* vector_values = vectors.data();
    const std::size_t values_per_case =
        static_cast<std::size_t>(expected_points) * 2;
    const int thread_count = std::max(1, requested_threads);
    std::vector<ExactRasterScratch> thread_scratch(
        static_cast<std::size_t>(thread_count));

    {
      py::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(thread_count)
#endif
      for (py::ssize_t case_index = 0; case_index < case_count; ++case_index) {
#ifdef _OPENMP
        const int thread_index = omp_get_thread_num();
#else
        const int thread_index = 0;
#endif
        const float* values = vector_values
            + static_cast<std::size_t>(case_index) * values_per_case;
        std::vector<Polygon> predicted(static_cast<std::size_t>(contour_count));
        for (int contour = 0; contour < contour_count; ++contour) {
          auto& polygon = predicted[static_cast<std::size_t>(contour)];
          polygon.reserve(static_cast<std::size_t>(anchors_per_contour));
          const int base = contour * anchors_per_contour * 2;
          for (int anchor = 0; anchor < anchors_per_contour; ++anchor) {
            const int offset = base + anchor * 2;
            polygon.emplace_back(values[offset], values[offset + 1]);
          }
        }
        const ExactMetricCounts metrics = exact_metric_counts_from_reference(
            exact_references_[static_cast<std::size_t>(frame_view(case_index))],
            predicted,
            &thread_scratch[static_cast<std::size_t>(thread_index)]);
        output_view(case_index, 0) = static_cast<double>(metrics.gt_area);
        output_view(case_index, 1) = static_cast<double>(metrics.pred_area);
        output_view(case_index, 2) = static_cast<double>(metrics.intersection);
        output_view(case_index, 3) = static_cast<double>(metrics.union_area);
        output_view(case_index, 4) = metrics.recall;
        output_view(case_index, 5) = metrics.precision;
        output_view(case_index, 6) = metrics.iou;
      }
    }
    return output;
  }

  py::array_t<double> evaluate_edge_batch(
      const py::array_t<float, py::array::c_style | py::array::forcecast>& candidate_vectors,
      const py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>& edges,
      const int contour_count,
      const int anchors_per_contour,
      const double iou_weight,
      const double recall_floor,
      const double normalization_scale,
      const double shape_update_threshold,
      const double shape_switch_weight,
      const double shape_distance_weight,
      const double adapt_gain,
      const double distance_relief,
      const double switch_relief,
      const double distance_min_scale,
      const double switch_min_scale,
      const int requested_threads,
      const py::object& precomputed_shape_distances,
      const bool short_circuit_infeasible,
      const bool skip_exact_recall,
      const bool exact_recall_first,
      const py::object& precomputed_recall_hint_frames) {
    if (candidate_vectors.ndim() != 4 || candidate_vectors.shape(3) != 2) {
      throw py::value_error("candidate_vectors must have shape (frames, states, points, 2)");
    }
    if (edges.ndim() != 2 || edges.shape(1) != 4) {
      throw py::value_error("edges must have shape (E, 4)");
    }
    const int frame_count_value = static_cast<int>(candidate_vectors.shape(0));
    const int state_count = static_cast<int>(candidate_vectors.shape(1));
    const int point_count = static_cast<int>(candidate_vectors.shape(2));
    if (frame_count_value != static_cast<int>(contexts_.size()) ||
        point_count != contour_count * anchors_per_contour) {
      throw py::value_error("candidate vector dimensions do not match evaluator topology");
    }
    const py::ssize_t edge_count = edges.shape(0);
    py::array_t<float, py::array::c_style | py::array::forcecast> shape_distances;
    const float* shape_distance_values = nullptr;
    if (!precomputed_shape_distances.is_none()) {
      shape_distances = py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(
          precomputed_shape_distances);
      if (!shape_distances || shape_distances.ndim() != 1 ||
          shape_distances.shape(0) != edge_count) {
        throw py::value_error("precomputed_shape_distances must have shape (E,)");
      }
      shape_distance_values = shape_distances.data();
    }
    py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>
        recall_hint_frames;
    const std::int32_t* recall_hint_values = nullptr;
    int recall_hint_count = 0;
    if (!precomputed_recall_hint_frames.is_none()) {
      recall_hint_frames =
          py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>::ensure(
              precomputed_recall_hint_frames);
      if (!recall_hint_frames ||
          (recall_hint_frames.ndim() != 1 && recall_hint_frames.ndim() != 2) ||
          recall_hint_frames.shape(0) != edge_count) {
        throw py::value_error(
            "precomputed_recall_hint_frames must have shape (E,) or (E, K)");
      }
      recall_hint_count = recall_hint_frames.ndim() == 1
          ? 1
          : static_cast<int>(recall_hint_frames.shape(1));
      if (recall_hint_count < 1 || recall_hint_count > 8) {
        throw py::value_error("recall hint count must be in [1, 8]");
      }
      recall_hint_values = recall_hint_frames.data();
    }
    py::array_t<double> output({edge_count, static_cast<py::ssize_t>(9)});
    auto output_view = output.mutable_unchecked<2>();
    const auto edge_view = edges.unchecked<2>();
    const float* vectors = candidate_vectors.data();
    const std::size_t values_per_state = static_cast<std::size_t>(point_count) * 2;
    const std::size_t values_per_frame =
        static_cast<std::size_t>(state_count) * values_per_state;
    const int thread_count = std::max(1, requested_threads);

    // One edge is evaluated at a time by each worker. Keeping a private mask
    // for every frame made scratch memory grow as threads * video length * ROI.
    // A worker only needs the current frame, so reuse a max-ROI allocation.
    int max_height = 1;
    int max_width = 1;
    for (const auto& context : contexts_) {
      max_height = std::max(max_height, context.gt_mask.rows);
      max_width = std::max(max_width, context.gt_mask.cols);
    }
    std::vector<cv::Mat> thread_pred(static_cast<std::size_t>(thread_count));
    std::vector<cv::Mat> thread_intersection(static_cast<std::size_t>(thread_count));
    std::vector<ExactRasterScratch> thread_exact(
        static_cast<std::size_t>(thread_count));
    for (int thread = 0; thread < thread_count; ++thread) {
      thread_pred[static_cast<std::size_t>(thread)] =
          cv::Mat::zeros(max_height, max_width, CV_8UC1);
      thread_intersection[static_cast<std::size_t>(thread)] =
          cv::Mat::zeros(max_height, max_width, CV_8UC1);
    }
    const int edge_schedule_chunk =
        edge_count <= static_cast<py::ssize_t>(thread_count * 16) ? 1 : 16;

    {
      py::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, edge_schedule_chunk) num_threads(thread_count)
#endif
      for (py::ssize_t edge_index = 0; edge_index < edge_count; ++edge_index) {
#ifdef _OPENMP
        const int thread_index = omp_get_thread_num();
#else
        const int thread_index = 0;
#endif
        const int start_frame = edge_view(edge_index, 0);
        const int start_state = edge_view(edge_index, 1);
        const int end_frame = edge_view(edge_index, 2);
        const int end_state = edge_view(edge_index, 3);
        const float* start = vectors
            + static_cast<std::size_t>(start_frame) * values_per_frame
            + static_cast<std::size_t>(start_state) * values_per_state;
        const float* end = vectors
            + static_cast<std::size_t>(end_frame) * values_per_frame
            + static_cast<std::size_t>(end_state) * values_per_state;
        const std::int32_t* recall_hint_row = recall_hint_values == nullptr
            ? nullptr
            : recall_hint_values + edge_index * recall_hint_count;
        double exact_recall_deficit = 0.0;
        if (exact_recall_first && !skip_exact_recall) {
          exact_recall_deficit = exact_interval_recall_deficit(
              start,
              end,
              contour_count,
              anchors_per_contour,
              start_frame,
              end_frame,
              false,
              recall_floor,
              recall_hint_row,
              recall_hint_count,
              &thread_exact[static_cast<std::size_t>(thread_index)]);
        }
        CachedMetricsTotals totals;
        if (exact_recall_deficit <= 1e-10) {
          totals = evaluate_cached_impl(
              start,
              end,
              contour_count,
              anchors_per_contour,
              start_frame,
              end_frame,
              false,
              iou_weight,
              recall_floor,
              &thread_pred[static_cast<std::size_t>(thread_index)],
              &thread_intersection[static_cast<std::size_t>(thread_index)],
              short_circuit_infeasible);
        }
        if (!exact_recall_first && !skip_exact_recall &&
            totals.recall_deficit_total <= 1e-10) {
          exact_recall_deficit = exact_interval_recall_deficit(
              start,
              end,
              contour_count,
              anchors_per_contour,
              start_frame,
              end_frame,
              false,
              recall_floor,
              recall_hint_row,
              recall_hint_count,
              &thread_exact[static_cast<std::size_t>(thread_index)]);
        }
        const double distance = shape_distance_values == nullptr
            ? shape_distance_impl(
                  start,
                  end,
                  static_cast<std::size_t>(point_count),
                  normalization_scale)
            : static_cast<double>(shape_distance_values[edge_index]);
        const double update = distance > shape_update_threshold ? 1.0 : 0.0;
        const double frame_loss_mean = totals.frame_loss_total /
            static_cast<double>(std::max(totals.frames_covered, 1));
        const double base = 1.0 + std::max(adapt_gain, 0.0) *
            std::max(frame_loss_mean, 0.0);
        const double distance_scale = std::max(
            distance_min_scale,
            1.0 / std::max(std::pow(base, std::max(distance_relief, 0.0)), 1e-6));
        const double switch_scale = std::max(
            switch_min_scale,
            1.0 / std::max(std::pow(base, std::max(switch_relief, 0.0)), 1e-6));
        const double cost = totals.frame_loss_total
            + shape_switch_weight * switch_scale * update
            + shape_distance_weight * distance_scale * distance;
        output_view(edge_index, 0) = cost;
        output_view(edge_index, 1) = distance;
        output_view(edge_index, 2) = update;
        output_view(edge_index, 3) = static_cast<double>(totals.frames_covered);
        output_view(edge_index, 4) = frame_loss_mean;
        output_view(edge_index, 5) = distance_scale;
        output_view(edge_index, 6) = switch_scale;
        output_view(edge_index, 7) = totals.recall_deficit_total;
        output_view(edge_index, 8) = exact_recall_deficit;
      }
    }
    return output;
  }

  [[nodiscard]] int frame_count() const {
    return static_cast<int>(contexts_.size());
  }

  py::dict context_statistics() const {
    py::dict result;
    if (contexts_.empty()) {
      result["frames"] = 0;
      result["min_height"] = 0;
      result["max_height"] = 0;
      result["mean_height"] = 0.0;
      result["min_width"] = 0;
      result["max_width"] = 0;
      result["mean_width"] = 0.0;
      result["mean_pixels"] = 0.0;
      result["max_pixels"] = 0;
      return result;
    }
    int min_height = std::numeric_limits<int>::max();
    int max_height = 0;
    int min_width = std::numeric_limits<int>::max();
    int max_width = 0;
    std::int64_t total_height = 0;
    std::int64_t total_width = 0;
    std::int64_t total_pixels = 0;
    std::int64_t max_pixels = 0;
    for (const auto& context : contexts_) {
      min_height = std::min(min_height, context.gt_mask.rows);
      max_height = std::max(max_height, context.gt_mask.rows);
      min_width = std::min(min_width, context.gt_mask.cols);
      max_width = std::max(max_width, context.gt_mask.cols);
      total_height += context.gt_mask.rows;
      total_width += context.gt_mask.cols;
      const std::int64_t pixels =
          static_cast<std::int64_t>(context.gt_mask.rows) * context.gt_mask.cols;
      total_pixels += pixels;
      max_pixels = std::max(max_pixels, pixels);
    }
    const double count = static_cast<double>(contexts_.size());
    result["frames"] = static_cast<int>(contexts_.size());
    result["min_height"] = min_height;
    result["max_height"] = max_height;
    result["mean_height"] = static_cast<double>(total_height) / count;
    result["min_width"] = min_width;
    result["max_width"] = max_width;
    result["mean_width"] = static_cast<double>(total_width) / count;
    result["mean_pixels"] = static_cast<double>(total_pixels) / count;
    result["max_pixels"] = max_pixels;
    return result;
  }

 private:
  double exact_interval_recall_deficit(
      const float* start,
      const float* end,
      const int contour_count,
      const int anchors_per_contour,
      const int start_index,
      const int end_index,
      const bool include_start,
      const double recall_floor,
      const std::int32_t* recall_hint_frames,
      const int recall_hint_count,
      ExactRasterScratch* scratch) const {
    const int first_index = include_start ? start_index : start_index + 1;
    std::vector<Polygon> predicted(static_cast<std::size_t>(contour_count));
    for (auto& polygon : predicted) {
      polygon.resize(static_cast<std::size_t>(anchors_per_contour));
    }
    std::array<int, 8> valid_recall_hints{};
    int valid_recall_hint_count = 0;
    for (int hint = 0; hint < recall_hint_count; ++hint) {
      const int frame = static_cast<int>(recall_hint_frames[hint]);
      if (frame < first_index || frame > end_index) {
        continue;
      }
      bool duplicate = false;
      for (int previous = 0; previous < valid_recall_hint_count; ++previous) {
        duplicate = duplicate || valid_recall_hints[previous] == frame;
      }
      if (!duplicate) {
        valid_recall_hints[valid_recall_hint_count++] = frame;
      }
    }
    const int frame_count_value = end_index - first_index + 1;
    int sequential_frame = first_index;
    for (int frame_order = 0; frame_order < frame_count_value; ++frame_order) {
      int frame_index = -1;
      if (frame_order < valid_recall_hint_count) {
        frame_index = valid_recall_hints[frame_order];
      } else {
        while (sequential_frame <= end_index) {
          bool is_hint = false;
          for (int hint = 0; hint < valid_recall_hint_count; ++hint) {
            is_hint = is_hint || valid_recall_hints[hint] == sequential_frame;
          }
          const int candidate = sequential_frame++;
          if (!is_hint) {
            frame_index = candidate;
            break;
          }
        }
      }
      const double alpha64 =
          static_cast<double>(frame_index - start_index) /
          static_cast<double>(std::max(end_index - start_index, 1));
      // Production's exact path converts alpha and (1 - alpha) to float32
      // independently. This differs by one float32 ULP from 1.0F-alpha for
      // ratios such as 1/3 and is observable at the hard Recall boundary.
      const float alpha = static_cast<float>(alpha64);
      const float beta = static_cast<float>(1.0 - alpha64);
      for (int contour = 0; contour < contour_count; ++contour) {
        const int base = contour * anchors_per_contour * 2;
        for (int anchor = 0; anchor < anchors_per_contour; ++anchor) {
          const int offset = base + anchor * 2;
          predicted[static_cast<std::size_t>(contour)][static_cast<std::size_t>(anchor)] =
              cv::Point2f(
                  beta * start[offset] + alpha * end[offset],
                  beta * start[offset + 1] + alpha * end[offset + 1]);
        }
      }
      const auto& reference =
          exact_references_[static_cast<std::size_t>(frame_index)];
      bool predicted_has_polygon = false;
      float predicted_min_x = std::numeric_limits<float>::infinity();
      float predicted_min_y = std::numeric_limits<float>::infinity();
      float predicted_max_x = -std::numeric_limits<float>::infinity();
      float predicted_max_y = -std::numeric_limits<float>::infinity();
      for (const auto& polygon : predicted) {
        if (polygon.size() < 3) {
          continue;
        }
        predicted_has_polygon = true;
        for (const auto& point : polygon) {
          predicted_min_x = std::min(predicted_min_x, point.x);
          predicted_min_y = std::min(predicted_min_y, point.y);
          predicted_max_x = std::max(predicted_max_x, point.x);
          predicted_max_y = std::max(predicted_max_y, point.y);
        }
      }
      if (!reference.has_polygon && !predicted_has_polygon) {
        continue;
      }
      if (!reference.has_polygon) {
        continue;
      }
      if (!predicted_has_polygon) {
        if (recall_floor > 1e-10) {
          return recall_floor;
        }
        continue;
      }
      const int joint_shift_x = static_cast<int>(std::floor(
          std::min(reference.min_x, predicted_min_x)));
      const int joint_shift_y = static_cast<int>(std::floor(
          std::min(reference.min_y, predicted_min_y)));
      const int parity_x = positive_modulo_two(joint_shift_x);
      const int parity_y = positive_modulo_two(joint_shift_y);
      const auto& variant = reference.variants[
          static_cast<std::size_t>(parity_x + 2 * parity_y)];
      const int pred_origin_x = origin_with_parity(
          static_cast<int>(std::floor(predicted_min_x)), parity_x);
      const int pred_origin_y = origin_with_parity(
          static_cast<int>(std::floor(predicted_min_y)), parity_y);
      const int pred_maximum_x = static_cast<int>(std::ceil(predicted_max_x));
      const int pred_maximum_y = static_cast<int>(std::ceil(predicted_max_y));
      const int pred_width = pred_maximum_x - pred_origin_x + 1;
      const int pred_height = pred_maximum_y - pred_origin_y + 1;
      const std::size_t pred_pixels =
          static_cast<std::size_t>(pred_width) *
          static_cast<std::size_t>(pred_height);
      if (scratch->prediction.size() < pred_pixels) {
        scratch->prediction.resize(pred_pixels);
      }
      cv::Mat pred_mask(
          pred_height,
          pred_width,
          CV_8UC1,
          scratch->prediction.data(),
          static_cast<std::size_t>(pred_width));
      rasterize_into(
          predicted,
          pred_mask,
          static_cast<float>(pred_origin_x),
          static_cast<float>(pred_origin_y));
      std::int64_t intersection = 0;
      for (const RasterRun& run : variant.runs) {
        const int local_y = run.y - pred_origin_y;
        if (local_y < 0 || local_y >= pred_height) {
          continue;
        }
        const int start_x = std::max(run.start_x, pred_origin_x);
        const int end_x = std::min(run.end_x, pred_origin_x + pred_width);
        if (start_x >= end_x) {
          continue;
        }
        const std::uint8_t* row = pred_mask.ptr<std::uint8_t>(local_y);
        for (int x = start_x - pred_origin_x; x < end_x - pred_origin_x; ++x) {
          intersection += static_cast<std::int64_t>(row[x]);
        }
      }
      const std::int64_t gt_area = variant.area;
      const double recall = gt_area > 0
          ? static_cast<double>(intersection) / gt_area
          : 1.0;
      const double deficit = std::max(recall_floor - recall, 0.0);
      if (deficit > 1e-10) {
        return deficit;
      }
    }
    return 0.0;
  }

  CachedMetricsTotals evaluate_cached_impl(
      const float* start,
      const float* end,
      const int contour_count,
      const int anchors_per_contour,
      const int start_index,
      const int end_index,
      const bool include_start,
      const double iou_weight,
      const double recall_floor,
      cv::Mat* external_pred,
      cv::Mat* external_intersection,
      const bool short_circuit_infeasible = false) {
    CachedMetricsTotals totals;
    std::vector<cv::Point> rounded(
        static_cast<std::size_t>(contour_count) *
        static_cast<std::size_t>(anchors_per_contour));
    const int first_index = include_start ? start_index : start_index + 1;
    for (int frame_index = first_index; frame_index <= end_index; ++frame_index) {
      CachedFrameContext& context = contexts_[static_cast<std::size_t>(frame_index)];
      cv::Mat pred_view;
      cv::Mat intersection_view;
      cv::Mat* pred_mask_ptr = &context.pred_mask;
      cv::Mat* intersection_mask_ptr = &context.intersection_mask;
      const float alpha = static_cast<float>(
          static_cast<double>(frame_index - start_index) /
          static_cast<double>(std::max(end_index - start_index, 1)));
      const float beta = 1.0F - alpha;
      int minimum_x = std::numeric_limits<int>::max();
      int minimum_y = std::numeric_limits<int>::max();
      int maximum_x = std::numeric_limits<int>::min();
      int maximum_y = std::numeric_limits<int>::min();
      for (int contour = 0; contour < contour_count; ++contour) {
        const int base = contour * anchors_per_contour * 2;
        for (int anchor = 0; anchor < anchors_per_contour; ++anchor) {
          const int offset = base + anchor * 2;
          const float mixed_x = beta * start[offset] + alpha * end[offset];
          const float mixed_y = beta * start[offset + 1] + alpha * end[offset + 1];
          const cv::Point point(
              static_cast<int>(std::nearbyint(
                  (mixed_x - context.shift_x) * context.scale_factor)),
              static_cast<int>(std::nearbyint(
                  (mixed_y - context.shift_y) * context.scale_factor)));
          rounded[static_cast<std::size_t>(contour * anchors_per_contour + anchor)] =
              point;
          minimum_x = std::min(minimum_x, point.x);
          minimum_y = std::min(minimum_y, point.y);
          maximum_x = std::max(maximum_x, point.x);
          maximum_y = std::max(maximum_y, point.y);
        }
      }
      cv::Rect compact_bounds;
      if (external_pred != nullptr && external_intersection != nullptr &&
          minimum_x <= maximum_x && minimum_y <= maximum_y) {
        const int clipped_minimum_x = std::max(0, minimum_x);
        const int clipped_minimum_y = std::max(0, minimum_y);
        const int clipped_maximum_x = std::min(context.gt_mask.cols - 1, maximum_x);
        const int clipped_maximum_y = std::min(context.gt_mask.rows - 1, maximum_y);
        if (clipped_minimum_x <= clipped_maximum_x &&
            clipped_minimum_y <= clipped_maximum_y) {
          compact_bounds = cv::Rect(
              clipped_minimum_x,
              clipped_minimum_y,
              clipped_maximum_x - clipped_minimum_x + 1,
              clipped_maximum_y - clipped_minimum_y + 1);
          // The max-sized thread buffers are byte arenas.  Giving the compact
          // view its own row stride avoids clearing and scanning empty parts
          // of a track-wide motion ROI while preserving integer fillPoly
          // semantics exactly under translation.
          pred_view = cv::Mat(
              compact_bounds.height,
              compact_bounds.width,
              CV_8UC1,
              external_pred->data,
              static_cast<std::size_t>(compact_bounds.width));
          intersection_view = cv::Mat(
              compact_bounds.height,
              compact_bounds.width,
              CV_8UC1,
              external_intersection->data,
              static_cast<std::size_t>(compact_bounds.width));
          pred_mask_ptr = &pred_view;
          intersection_mask_ptr = &intersection_view;
        }
      }
      if (external_pred != nullptr && external_intersection != nullptr &&
          compact_bounds.empty()) {
        // Keep batch workers on their private storage even when the predicted
        // polygon is completely outside the cached context.  A one-pixel
        // clipped view produces the same zero foreground/intersection as the
        // historical full-context raster without sharing mutable context
        // buffers across OpenMP workers.
        compact_bounds = cv::Rect(0, 0, 1, 1);
        pred_view = cv::Mat(
            1,
            1,
            CV_8UC1,
            external_pred->data,
            static_cast<std::size_t>(1));
        intersection_view = cv::Mat(
            1,
            1,
            CV_8UC1,
            external_intersection->data,
            static_cast<std::size_t>(1));
        pred_mask_ptr = &pred_view;
        intersection_mask_ptr = &intersection_view;
      }
      cv::Mat& pred_mask = *pred_mask_ptr;
      cv::Mat& intersection_mask = *intersection_mask_ptr;
      pred_mask.setTo(cv::Scalar(0));
      for (int contour = 0; contour < contour_count; ++contour) {
        cv::Point* points = rounded.data() + contour * anchors_per_contour;
        if (!compact_bounds.empty()) {
          for (int anchor = 0; anchor < anchors_per_contour; ++anchor) {
            points[anchor].x -= compact_bounds.x;
            points[anchor].y -= compact_bounds.y;
          }
        }
        if (anchors_per_contour >= 3) {
          const cv::Point* polygon = points;
          const int point_count_local = anchors_per_contour;
          cv::fillPoly(
              pred_mask,
              &polygon,
              &point_count_local,
              1,
              cv::Scalar(1));
        }
      }
      const std::int64_t pred_area = cv::countNonZero(pred_mask);
      if (!compact_bounds.empty()) {
        cv::bitwise_and(
            context.gt_mask(compact_bounds), pred_mask, intersection_mask);
      } else {
        cv::bitwise_and(context.gt_mask, pred_mask, intersection_mask);
      }
      const std::int64_t intersection = cv::countNonZero(intersection_mask);
      const std::int64_t union_area = context.gt_area + pred_area - intersection;
      const double recall = context.gt_area > 0
          ? static_cast<double>(intersection) / context.gt_area
          : 1.0;
      const double iou = union_area > 0
          ? static_cast<double>(intersection) / union_area
          : 1.0;
      totals.frame_loss_total += iou_weight * (1.0 - iou);
      totals.recall_deficit_total += std::max(recall_floor - recall, 0.0);
      ++totals.frames_covered;
      // Batch DP callers only need to know whether an edge is feasible.  Once
      // one frame violates the hard Recall floor, the edge cost is discarded
      // and the remaining frames cannot change that decision.  Avoiding those
      // rasterizations is especially important for long, high-resolution
      // tracks.  The scalar evaluator keeps the historical full-interval
      // totals by leaving this option disabled.
      if (short_circuit_infeasible && totals.recall_deficit_total > 1e-10) {
        break;
      }
    }
    return totals;
  }

  std::vector<CachedFrameContext> contexts_;
  std::vector<ExactFrameReference> exact_references_;
  ExactRasterScratch endpoint_exact_scratch_;
};

}  // namespace

PYBIND11_MODULE(native_interval_metrics, module) {
  module.doc() = "Exact polygon raster metrics for the Production runtime";
  module.def(
      "exact_metrics",
      &exact_metrics,
      py::arg("gt_polygons"),
      py::arg("pred_polygons"));
  module.def(
      "pair_vote_local_metrics_batch",
      &pair_vote_local_metrics_batch,
      py::arg("gt_frames"),
      py::arg("chosen_frames"),
      py::arg("current_vectors"),
      py::arg("key_pos"),
      py::arg("trial_vectors"),
      py::arg("contour_count"),
      py::arg("anchors_per_contour"),
      py::arg("threads") = 1);
  module.def(
      "pair_vote_full_metrics_batch",
      &pair_vote_full_metrics_batch,
      py::arg("gt_frames"),
      py::arg("chosen_frames"),
      py::arg("trial_key_vectors"),
      py::arg("contour_count"),
      py::arg("anchors_per_contour"),
      py::arg("threads") = 1);
  module.def(
      "canonical_metrics",
      &canonical_metrics,
      py::arg("gt_polygons"),
      py::arg("pred_polygons"));
  module.def(
      "shape_distance",
      &shape_distance,
      py::arg("source"),
      py::arg("destination"),
      py::arg("normalization_scale"));
  module.def(
      "strict_self_intersection_batch",
      &strict_self_intersection_batch,
      py::arg("values"),
      py::arg("threads") = 1);
  module.def(
      "polygon_is_simple_batch",
      &polygon_is_simple_batch,
      py::arg("values"),
      py::arg("threads") = 1);
  module.def(
      "shared_area_best_cycle",
      &shared_area_best_cycle,
      py::arg("costs"),
      py::arg("target"));
  module.def(
      "decode_penalty_path",
      &decode_penalty_path,
      py::arg("edges"),
      py::arg("edge_costs"),
      py::arg("initial_losses"),
      py::arg("frame_count"),
      py::arg("state_count"),
      py::arg("penalty"));
  py::class_<IncrementalPenaltyPathDecoder>(
      module, "IncrementalPenaltyPathDecoder")
      .def(
          py::init<
              const py::array_t<
                  std::int32_t,
                  py::array::c_style | py::array::forcecast>&,
              const py::array_t<
                  double,
                  py::array::c_style | py::array::forcecast>&,
              int,
              int>(),
          py::arg("edges"),
          py::arg("initial_losses"),
          py::arg("frame_count"),
          py::arg("state_count"))
      .def(
          "decode",
          &IncrementalPenaltyPathDecoder::decode,
          py::arg("edge_costs"),
          py::arg("penalty"),
          py::arg("recompute_from") = 0);
  py::class_<ExactDoubleRasterEvaluator>(module, "ExactDoubleRasterEvaluator")
      .def(
          py::init<const py::iterable&, std::uint64_t>(),
          py::arg("reference_frames"),
          py::arg("maximum_cache_bytes") = 256U * 1024U * 1024U)
      .def("cache_stats", &ExactDoubleRasterEvaluator::cache_stats)
      .def(
          "metrics_batch",
          &ExactDoubleRasterEvaluator::metrics_batch,
          py::arg("frame_indices"),
          py::arg("vectors"),
          py::arg("contour_count"),
          py::arg("anchors_per_contour"),
          py::arg("threads") = 1)
      .def(
          "catmull_metrics_batch",
          &ExactDoubleRasterEvaluator::catmull_metrics_batch,
          py::arg("frame_indices"),
          py::arg("controls"),
          py::arg("sampling_matrix"),
          py::arg("threads") = 1,
          py::arg("check_topology") = false)
      .def(
          "catmull_scale_metrics_batch",
          &ExactDoubleRasterEvaluator::catmull_scale_metrics_batch,
          py::arg("controls"),
          py::arg("scales"),
          py::arg("sampling_matrix"),
          py::arg("threads") = 1,
          py::arg("check_topology") = true)
      .def(
          "catmull_topology_batch",
          &ExactDoubleRasterEvaluator::catmull_topology_batch,
          py::arg("controls"),
          py::arg("sampling_matrix"),
          py::arg("threads") = 1)
      .def(
          "edge_metrics_batch",
          &ExactDoubleRasterEvaluator::edge_metrics_batch,
          py::arg("candidate_boundaries"),
          py::arg("edges"),
          py::arg("recall_floor"),
          py::arg("low_iou_quadratic_weight"),
          py::arg("threads") = 1,
          py::arg("check_topology") = true);
  py::class_<CachedIntervalEvaluator>(module, "CachedIntervalEvaluator")
      .def(
          py::init<
              const py::iterable&,
              const py::array_t<float, py::array::c_style | py::array::forcecast>&,
              const py::array_t<float, py::array::c_style | py::array::forcecast>&,
              const py::iterable&>(),
          py::arg("gt_masks"),
          py::arg("shifts"),
          py::arg("scales"),
          py::arg("exact_gt_frames"))
      .def(
          "evaluate_vectors",
          &CachedIntervalEvaluator::evaluate_vectors,
          py::arg("start_vector"),
          py::arg("end_vector"),
          py::arg("contour_count"),
          py::arg("anchors_per_contour"),
          py::arg("start_index"),
          py::arg("end_index"),
          py::arg("include_start"),
          py::arg("iou_weight"),
          py::arg("recall_floor"))
      .def(
          "evaluate_edge_batch",
          &CachedIntervalEvaluator::evaluate_edge_batch,
          py::arg("candidate_vectors"),
          py::arg("edges"),
          py::arg("contour_count"),
          py::arg("anchors_per_contour"),
          py::arg("iou_weight"),
          py::arg("recall_floor"),
          py::arg("normalization_scale"),
          py::arg("shape_update_threshold"),
          py::arg("shape_switch_weight"),
          py::arg("shape_distance_weight"),
          py::arg("adapt_gain"),
          py::arg("distance_relief"),
          py::arg("switch_relief"),
          py::arg("distance_min_scale"),
          py::arg("switch_min_scale"),
          py::arg("threads"),
          py::arg("precomputed_shape_distances") = py::none(),
          py::arg("short_circuit_infeasible") = false,
          py::arg("skip_exact_recall") = false,
          py::arg("exact_recall_first") = false,
          py::arg("precomputed_recall_hint_frames") = py::none())
      .def(
          "exact_frame_metrics",
          &CachedIntervalEvaluator::exact_frame_metrics,
          py::arg("frame_index"),
          py::arg("vector"),
          py::arg("contour_count"),
          py::arg("anchors_per_contour"))
      .def(
          "exact_frame_metrics_batch",
          &CachedIntervalEvaluator::exact_frame_metrics_batch,
          py::arg("frame_indices"),
          py::arg("vectors"),
          py::arg("contour_count"),
          py::arg("anchors_per_contour"),
          py::arg("threads") = 1)
      .def_property_readonly("frame_count", &CachedIntervalEvaluator::frame_count)
      .def("context_statistics", &CachedIntervalEvaluator::context_statistics);
  module.attr("implementation") = "cpp-opencv-4.8";
}
