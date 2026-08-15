#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

struct Point {
  double x{};
  double y{};
};

double cross(const Point& a, const Point& b) { return a.x * b.y - a.y * b.x; }

double squared_distance(const Point& a, const Point& b) {
  const double dx = a.x - b.x;
  const double dy = a.y - b.y;
  return dx * dx + dy * dy;
}

double point_segment_squared(const Point& p, const Point& a, const Point& b) {
  const double dx = b.x - a.x;
  const double dy = b.y - a.y;
  const double denominator = dx * dx + dy * dy;
  if (denominator <= 1e-18) return squared_distance(p, a);
  const double alpha = std::clamp(((p.x - a.x) * dx + (p.y - a.y) * dy) / denominator, 0.0, 1.0);
  const Point projection{a.x + alpha * dx, a.y + alpha * dy};
  return squared_distance(p, projection);
}

std::vector<Point> array_to_points(const py::handle& value) {
  auto array = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(value);
  if (!array || array.ndim() != 2 || array.shape(1) != 2 || array.shape(0) < 3) {
    throw std::invalid_argument("each contour must have shape (N, 2), N >= 3");
  }
  auto input = array.unchecked<2>();
  std::vector<Point> points(static_cast<std::size_t>(array.shape(0)));
  for (py::ssize_t index = 0; index < array.shape(0); ++index) {
    points[static_cast<std::size_t>(index)] = Point{input(index, 0), input(index, 1)};
  }
  return points;
}

double signed_area(const std::vector<Point>& points) {
  double twice = 0.0;
  for (std::size_t index = 0; index < points.size(); ++index) {
    twice += cross(points[index], points[(index + 1) % points.size()]);
  }
  return 0.5 * twice;
}

std::vector<Point> resample_closed(const std::vector<Point>& points, std::size_t count) {
  if (points.empty() || count < 3) throw std::invalid_argument("invalid contour resample");
  std::vector<double> lengths(points.size(), 0.0);
  double total = 0.0;
  for (std::size_t index = 0; index < points.size(); ++index) {
    lengths[index] = std::sqrt(squared_distance(points[index], points[(index + 1) % points.size()]));
    total += lengths[index];
  }
  if (total <= 1e-12) return std::vector<Point>(count, points.front());
  std::vector<Point> output(count);
  std::size_t segment = 0;
  double segment_start = 0.0;
  for (std::size_t sample = 0; sample < count; ++sample) {
    const double position = total * static_cast<double>(sample) / static_cast<double>(count);
    while (segment + 1 < points.size() && position >= segment_start + lengths[segment]) {
      segment_start += lengths[segment];
      ++segment;
    }
    const double alpha = (position - segment_start) / std::max(lengths[segment], 1e-12);
    const auto& left = points[segment];
    const auto& right = points[(segment + 1) % points.size()];
    output[sample] = Point{
        (1.0 - alpha) * left.x + alpha * right.x,
        (1.0 - alpha) * left.y + alpha * right.y,
    };
  }
  return output;
}

std::vector<Point> rotate_contour(const std::vector<Point>& input, std::size_t anchor) {
  std::vector<Point> output(input.size() + 1);
  for (std::size_t index = 0; index < input.size(); ++index) {
    output[index] = input[(anchor + index) % input.size()];
  }
  output.back() = output.front();
  return output;
}

std::vector<Point> fit_similarity_and_predict(
    const std::vector<Point>& previous_seed,
    const std::vector<Point>& current_seed,
    const std::vector<Point>& previous_output) {
  const std::size_t count = previous_seed.size();
  Point left_center{}, right_center{};
  for (std::size_t index = 0; index < count; ++index) {
    left_center.x += previous_seed[index].x;
    left_center.y += previous_seed[index].y;
    right_center.x += current_seed[index].x;
    right_center.y += current_seed[index].y;
  }
  left_center.x /= static_cast<double>(count);
  left_center.y /= static_cast<double>(count);
  right_center.x /= static_cast<double>(count);
  right_center.y /= static_cast<double>(count);
  double dot = 0.0;
  double determinant = 0.0;
  double denominator = 0.0;
  for (std::size_t index = 0; index < count; ++index) {
    const double lx = previous_seed[index].x - left_center.x;
    const double ly = previous_seed[index].y - left_center.y;
    const double rx = current_seed[index].x - right_center.x;
    const double ry = current_seed[index].y - right_center.y;
    dot += lx * rx + ly * ry;
    determinant += lx * ry - ly * rx;
    denominator += lx * lx + ly * ly;
  }
  const double a = dot / std::max(denominator, 1e-18);
  const double b = determinant / std::max(denominator, 1e-18);
  std::vector<Point> prediction(previous_output.size());
  for (std::size_t index = 0; index < previous_output.size(); ++index) {
    const double x = previous_output[index].x - left_center.x;
    const double y = previous_output[index].y - left_center.y;
    prediction[index] = Point{
        a * x - b * y + right_center.x,
        b * x + a * y + right_center.y,
    };
  }
  return prediction;
}

std::vector<Point> simplify_one(
    const std::vector<Point>& source,
    const std::vector<Point>& prediction,
    const std::vector<Point>& identity_seed,
    double temporal_weight,
    double distance_weight,
    double missing_area_weight,
    double excess_area_weight,
    double contour_band_fraction) {
  const std::size_t n = source.size();
  const std::size_t target = prediction.size();
  std::size_t anchor = 0;
  double anchor_distance = std::numeric_limits<double>::infinity();
  for (std::size_t index = 0; index < n; ++index) {
    const auto& anchor_target = contour_band_fraction > 0.0
        ? identity_seed[0]
        : prediction[0];
    const double distance = squared_distance(source[index], anchor_target);
    if (distance < anchor_distance) {
      anchor_distance = distance;
      anchor = index;
    }
  }
  const auto contour = rotate_contour(source, anchor);
  const double area = std::max(std::abs(signed_area(source)), 1.0);
  const double radius_squared = std::max(area / 3.14159265358979323846, 1.0);

  std::vector<double> prefix_cross(n + 1, 0.0);
  for (std::size_t index = 0; index < n; ++index) {
    prefix_cross[index + 1] = prefix_cross[index] + cross(contour[index], contour[index + 1]);
  }
  const std::size_t stride = n + 1;

  // Optional persistent-identity band.  The old Production implementation
  // assigns identities by equal arclength and phase alignment.  A completely
  // free per-frame simplifier can subsequently let a vertex jump to a remote
  // corner.  Here each predicted identity is first projected to the current
  // contour in cyclic order and may only slide within a fraction of one
  // average vertex spacing.  Zero keeps the original unrestricted DP.
  std::vector<std::size_t> expected(target, 0);
  std::size_t contour_band = n;
  if (contour_band_fraction > 0.0) {
    contour_band = std::max<std::size_t>(
        1,
        static_cast<std::size_t>(std::ceil(
            contour_band_fraction * static_cast<double>(n) /
            static_cast<double>(target))));
    // The identity seed is an already phase-aligned point on the current
    // contour (Production equal-arc or a shared track-level allocation).
    // Project it in cyclic order, then let the optimizer slide locally.
    expected[0] = 0;
    for (std::size_t position = 1; position < target; ++position) {
      const std::size_t lower = expected[position - 1] + 1;
      const std::size_t upper = n - (target - position);
      std::size_t best_index = lower;
      double best_distance = std::numeric_limits<double>::infinity();
      for (std::size_t index = lower; index <= upper; ++index) {
        const double value = squared_distance(contour[index], identity_seed[position]);
        if (value < best_distance) {
          best_distance = value;
          best_index = index;
        }
      }
      expected[position] = best_index;
    }
  }
  auto inside_band = [&](std::size_t position, std::size_t contour_index) {
    if (contour_band_fraction <= 0.0 || position == 0) return true;
    const std::size_t left = expected[position] > contour_band
        ? expected[position] - contour_band
        : 0;
    const std::size_t right = std::min(n - 1, expected[position] + contour_band);
    return contour_index >= left && contour_index <= right;
  };

  std::vector<double> edge(stride * stride, std::numeric_limits<double>::infinity());
  for (std::size_t first = 0; first < n; ++first) {
    edge[first * stride + first + 1] = 0.0;
    for (std::size_t last = first + 2; last <= n; ++last) {
      const double local_signed = 0.5 * (
          prefix_cross[last] - prefix_cross[first] + cross(contour[last], contour[first]));
      double maximum_squared = 0.0;
      for (std::size_t middle = first + 1; middle < last; ++middle) {
        maximum_squared = std::max(
            maximum_squared,
            point_segment_squared(contour[middle], contour[first], contour[last]));
      }
      const double area_cost = local_signed >= 0.0
          ? missing_area_weight * local_signed / area
          : excess_area_weight * (-local_signed) / area;
      edge[first * stride + last] = area_cost + distance_weight * maximum_squared / radius_squared;
    }
  }

  auto node_cost = [&](std::size_t position, std::size_t contour_index) {
    return temporal_weight * squared_distance(contour[contour_index], prediction[position]) / radius_squared;
  };
  const double infinity = std::numeric_limits<double>::infinity();
  std::vector<double> previous(stride, infinity), current(stride, infinity);
  std::vector<int> parents(target * stride, -1);
  previous[0] = node_cost(0, 0);
  for (std::size_t selected = 2; selected <= target; ++selected) {
    std::fill(current.begin(), current.end(), infinity);
    const std::size_t position = selected - 1;
    const std::size_t minimum = selected - 1;
    const std::size_t maximum = n - (target - selected) - 1;
    for (std::size_t last = minimum; last <= maximum; ++last) {
      if (!inside_band(position, last)) continue;
      const std::size_t previous_minimum = selected - 2;
      for (std::size_t first = previous_minimum; first < last; ++first) {
        if (!std::isfinite(previous[first])) continue;
        const double candidate = previous[first] + edge[first * stride + last];
        if (candidate < current[last]) {
          current[last] = candidate;
          parents[position * stride + last] = static_cast<int>(first);
        }
      }
      current[last] += node_cost(position, last);
    }
    previous.swap(current);
  }
  double best = infinity;
  int last_best = -1;
  for (std::size_t last = target - 1; last < n; ++last) {
    const double candidate = previous[last] + edge[last * stride + n];
    if (candidate < best) {
      best = candidate;
      last_best = static_cast<int>(last);
    }
  }
  if (last_best < 0) throw std::runtime_error("temporal polygon DP has no solution");
  std::vector<int> indices(target, 0);
  indices[target - 1] = last_best;
  for (std::size_t position = target - 1; position > 0; --position) {
    const int parent = parents[position * stride + static_cast<std::size_t>(indices[position])];
    if (parent < 0) throw std::runtime_error("broken temporal polygon DP parent chain");
    indices[position - 1] = parent;
  }
  std::vector<Point> output(target);
  for (std::size_t position = 0; position < target; ++position) {
    output[position] = contour[static_cast<std::size_t>(indices[position])];
  }
  return output;
}

struct RdpSegment {
  double error{};
  std::size_t start{};
  std::size_t end{};
  std::size_t split{};
  bool valid{};
  bool operator<(const RdpSegment& other) const {
    if (error != other.error) return error < other.error;
    if (start != other.start) return start > other.start;
    return end > other.end;
  }
};

RdpSegment rdp_segment(const std::vector<Point>& points, std::size_t start, std::size_t end) {
  const std::size_t count = points.size();
  RdpSegment result{0.0, start, end, 0, false};
  for (std::size_t index = (start + 1) % count; index != end; index = (index + 1) % count) {
    const double error = point_segment_squared(points[index], points[start], points[end]);
    if (!result.valid || error > result.error) {
      result.error = error;
      result.split = index;
      result.valid = true;
    }
  }
  return result;
}

std::vector<Point> rdp_fixed_count(const std::vector<Point>& points, std::size_t target) {
  const std::vector<Point> dense = points.size() < target
      ? resample_closed(points, std::max(target * 4, static_cast<std::size_t>(32)))
      : points;
  const auto& contour = dense;
  const std::size_t count = contour.size();
  target = std::clamp(target, static_cast<std::size_t>(3), count);
  if (target == count) return contour;
  std::size_t first = 0, second = 1;
  double diameter = -1.0;
  for (std::size_t left = 0; left < count; ++left) {
    for (std::size_t right = left + 1; right < count; ++right) {
      const double value = squared_distance(contour[left], contour[right]);
      if (value > diameter) {
        diameter = value;
        first = left;
        second = right;
      }
    }
  }
  std::vector<unsigned char> selected(count, 0);
  selected[first] = 1;
  selected[second] = 1;
  std::priority_queue<RdpSegment> queue;
  auto push = [&](std::size_t start, std::size_t end) {
    const auto segment = rdp_segment(contour, start, end);
    if (segment.valid) queue.push(segment);
  };
  push(first, second);
  push(second, first);
  std::size_t selected_count = 2;
  while (selected_count < target && !queue.empty()) {
    const auto segment = queue.top();
    queue.pop();
    if (selected[segment.split]) continue;
    selected[segment.split] = 1;
    ++selected_count;
    push(segment.start, segment.split);
    push(segment.split, segment.end);
  }
  std::vector<Point> output;
  output.reserve(target);
  for (std::size_t index = 0; index < count; ++index) {
    if (selected[index]) output.push_back(contour[index]);
  }
  if (output.size() != target) throw std::runtime_error("native RDP count mismatch");
  return output;
}

std::vector<Point> align_translation_phase(
    const std::vector<Point>& reference,
    const std::vector<Point>& candidate) {
  const std::size_t count = reference.size();
  Point reference_center{}, candidate_center{};
  for (std::size_t index = 0; index < count; ++index) {
    reference_center.x += reference[index].x;
    reference_center.y += reference[index].y;
    candidate_center.x += candidate[index].x;
    candidate_center.y += candidate[index].y;
  }
  reference_center.x /= static_cast<double>(count);
  reference_center.y /= static_cast<double>(count);
  candidate_center.x /= static_cast<double>(count);
  candidate_center.y /= static_cast<double>(count);
  std::size_t best_shift = 0;
  double best_cost = std::numeric_limits<double>::infinity();
  for (std::size_t shift = 0; shift < count; ++shift) {
    double cost = 0.0;
    for (std::size_t index = 0; index < count; ++index) {
      const auto& left = reference[index];
      const auto& right = candidate[(index + shift) % count];
      const double dx = (left.x - reference_center.x) - (right.x - candidate_center.x);
      const double dy = (left.y - reference_center.y) - (right.y - candidate_center.y);
      cost += dx * dx + dy * dy;
    }
    if (cost < best_cost) {
      best_cost = cost;
      best_shift = shift;
    }
  }
  std::vector<Point> output(count);
  for (std::size_t index = 0; index < count; ++index) {
    output[index] = candidate[(index + best_shift) % count];
  }
  return output;
}

py::array_t<double> build_rdp_seed(const py::list& contour_values, std::size_t target) {
  const std::size_t frames = static_cast<std::size_t>(py::len(contour_values));
  if (frames == 0 || target < 3) throw std::invalid_argument("invalid frames or target");
  std::vector<std::vector<Point>> seed(frames);
  for (std::size_t frame = 0; frame < frames; ++frame) {
    auto contour = array_to_points(contour_values[frame]);
    if (signed_area(contour) < 0.0) std::reverse(contour.begin(), contour.end());
    seed[frame] = rdp_fixed_count(contour, target);
  }
  const std::size_t center = frames / 2;
  for (std::size_t frame = center + 1; frame < frames; ++frame) {
    seed[frame] = align_translation_phase(seed[frame - 1], seed[frame]);
  }
  for (std::size_t frame = center; frame-- > 0;) {
    seed[frame] = align_translation_phase(seed[frame + 1], seed[frame]);
  }
  py::array_t<double> result({
      static_cast<py::ssize_t>(frames),
      static_cast<py::ssize_t>(target),
      static_cast<py::ssize_t>(2),
  });
  auto writable = result.mutable_unchecked<3>();
  for (std::size_t frame = 0; frame < frames; ++frame) {
    for (std::size_t position = 0; position < target; ++position) {
      writable(frame, position, 0) = seed[frame][position].x;
      writable(frame, position, 1) = seed[frame][position].y;
    }
  }
  return result;
}

py::array_t<double> simplify_sequence(
    const py::list& contour_values,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& seed_values,
    double temporal_weight,
    double distance_weight,
    double missing_area_weight,
    double excess_area_weight,
    double contour_band_fraction) {
  if (seed_values.ndim() != 3 || seed_values.shape(2) != 2) {
    throw std::invalid_argument("seed must have shape (T, K, 2)");
  }
  const std::size_t frames = static_cast<std::size_t>(seed_values.shape(0));
  const std::size_t target = static_cast<std::size_t>(seed_values.shape(1));
  if (static_cast<std::size_t>(py::len(contour_values)) != frames || target < 3) {
    throw std::invalid_argument("contour and seed dimensions disagree");
  }
  std::vector<std::vector<Point>> contours(frames), seeds(frames), output(frames);
  auto seed = seed_values.unchecked<3>();
  for (std::size_t frame = 0; frame < frames; ++frame) {
    contours[frame] = array_to_points(contour_values[frame]);
    if (signed_area(contours[frame]) < 0.0) std::reverse(contours[frame].begin(), contours[frame].end());
    if (contours[frame].size() <= target) {
      contours[frame] = resample_closed(
          contours[frame], std::max(target * 4, static_cast<std::size_t>(32)));
    }
    seeds[frame].resize(target);
    for (std::size_t position = 0; position < target; ++position) {
      seeds[frame][position] = Point{
          seed(static_cast<py::ssize_t>(frame), static_cast<py::ssize_t>(position), 0),
          seed(static_cast<py::ssize_t>(frame), static_cast<py::ssize_t>(position), 1),
      };
    }
  }
  const std::size_t center = frames / 2;
  output[center] = simplify_one(
      contours[center], seeds[center], seeds[center], temporal_weight, distance_weight,
      missing_area_weight, excess_area_weight, contour_band_fraction);
  for (std::size_t frame = center + 1; frame < frames; ++frame) {
    const auto prediction = fit_similarity_and_predict(seeds[frame - 1], seeds[frame], output[frame - 1]);
    output[frame] = simplify_one(
        contours[frame], prediction, seeds[frame], temporal_weight, distance_weight,
        missing_area_weight, excess_area_weight, contour_band_fraction);
  }
  for (std::size_t frame = center; frame-- > 0;) {
    const auto prediction = fit_similarity_and_predict(seeds[frame + 1], seeds[frame], output[frame + 1]);
    output[frame] = simplify_one(
        contours[frame], prediction, seeds[frame], temporal_weight, distance_weight,
        missing_area_weight, excess_area_weight, contour_band_fraction);
  }
  py::array_t<double> result({
      static_cast<py::ssize_t>(frames),
      static_cast<py::ssize_t>(target),
      static_cast<py::ssize_t>(2),
  });
  auto writable = result.mutable_unchecked<3>();
  for (std::size_t frame = 0; frame < frames; ++frame) {
    for (std::size_t position = 0; position < target; ++position) {
      writable(frame, position, 0) = output[frame][position].x;
      writable(frame, position, 1) = output[frame][position].y;
    }
  }
  return result;
}

py::array_t<double> simplify_sequence_auto(
    const py::list& contour_values,
    std::size_t target,
    double temporal_weight,
    double distance_weight,
    double missing_area_weight,
    double excess_area_weight,
    double contour_band_fraction) {
  auto seed = build_rdp_seed(contour_values, target);
  return simplify_sequence(
      contour_values, seed, temporal_weight, distance_weight,
      missing_area_weight, excess_area_weight, contour_band_fraction);
}

}  // namespace

PYBIND11_MODULE(native_temporal_polygon, module) {
  module.doc() = "Native fixed-count temporal polygon simplification";
  module.def(
      "simplify_sequence",
      &simplify_sequence,
      py::arg("contours"),
      py::arg("seed"),
      py::arg("temporal_weight") = 0.1,
      py::arg("distance_weight") = 1.0,
      py::arg("missing_area_weight") = 4.0,
      py::arg("excess_area_weight") = 1.0,
      py::arg("contour_band_fraction") = 0.0);
  module.def(
      "build_rdp_seed",
      &build_rdp_seed,
      py::arg("contours"),
      py::arg("target"));
  module.def(
      "simplify_sequence_auto",
      &simplify_sequence_auto,
      py::arg("contours"),
      py::arg("target"),
      py::arg("temporal_weight") = 0.003,
      py::arg("distance_weight") = 2.0,
      py::arg("missing_area_weight") = 1.0,
      py::arg("excess_area_weight") = 1.0,
      py::arg("contour_band_fraction") = 0.0);
}
