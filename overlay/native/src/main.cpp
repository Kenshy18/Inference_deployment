#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "cuda_overlay.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/error.h>
#include <libavutil/hwcontext.h>
#include <libavutil/opt.h>
#include <sqlite3.h>
}

namespace fs = std::filesystem;

namespace {

struct Point {
    double x{};
    double y{};
};

using Polygon = std::vector<Point>;

struct Box {
    double x1{};
    double y1{};
    double x2{};
    double y2{};
};

enum class ItemKind {
    mask,
    face,
};

struct Mask {
    ItemKind kind{ItemKind::mask};
    std::string track_id;
    std::string label;
    std::optional<double> score;
    std::vector<Polygon> polygons;
    std::optional<Box> box;
    std::array<std::uint8_t, 3> bgr{};
};

using FrameMasks = std::map<int, std::vector<Mask>>;

struct Options {
    fs::path video;
    fs::path sqlite;
    fs::path face_sqlite;
    fs::path output;
    fs::path manifest;
    std::string mode{"final"};
    std::string encoder{"h264_nvenc"};
    std::string preset{"p1"};
    double bitrate_mbps{8.0};
    std::optional<int> crf;
    double mask_alpha{0.32};
    int outline_thickness{2};
    int box_thickness{2};
    int decoder_threads{0};
    int gpu_index{0};
    bool include_faces{false};
    bool show_labels{true};
    bool copy_audio{false};
    bool hw_decode{false};
    bool gpu_pipeline{false};
    bool faststart{false};
    int start_frame{0};
    std::optional<int> end_frame;
    bool overwrite{false};
};

std::string av_error(int code) {
    std::array<char, AV_ERROR_MAX_STRING_SIZE> buffer{};
    av_strerror(code, buffer.data(), buffer.size());
    return std::string(buffer.data());
}

void check_av(int code, std::string_view operation) {
    if (code < 0) {
        throw std::runtime_error(
            std::string(operation) + ": " + av_error(code)
        );
    }
}

class JsonPolygonParser {
public:
    explicit JsonPolygonParser(std::string_view source)
        : current_(source.data()), end_(source.data() + source.size()) {}

    std::vector<Polygon> parse() {
        std::vector<Polygon> polygons;
        expect('[');
        skip_space();
        if (consume(']')) {
            return polygons;
        }
        while (true) {
            polygons.push_back(parse_polygon());
            skip_space();
            if (consume(']')) {
                break;
            }
            expect(',');
        }
        skip_space();
        if (current_ != end_) {
            fail("trailing JSON content");
        }
        polygons.erase(
            std::remove_if(
                polygons.begin(),
                polygons.end(),
                [](const Polygon& polygon) { return polygon.size() < 3; }
            ),
            polygons.end()
        );
        return polygons;
    }

private:
    Polygon parse_polygon() {
        Polygon polygon;
        expect('[');
        skip_space();
        if (consume(']')) {
            return polygon;
        }
        while (true) {
            polygon.push_back(parse_point());
            skip_space();
            if (consume(']')) {
                break;
            }
            expect(',');
        }
        return polygon;
    }

    Point parse_point() {
        expect('[');
        const double x = parse_number();
        expect(',');
        const double y = parse_number();
        skip_space();
        while (consume(',')) {
            static_cast<void>(parse_number());
            skip_space();
        }
        expect(']');
        return Point{x, y};
    }

    double parse_number() {
        skip_space();
        if (current_ == end_) {
            fail("expected number");
        }
        char* parsed_end = nullptr;
        const double value = std::strtod(current_, &parsed_end);
        if (parsed_end == current_ || parsed_end > end_) {
            fail("invalid number");
        }
        current_ = parsed_end;
        if (!std::isfinite(value)) {
            fail("non-finite number");
        }
        return value;
    }

    void skip_space() {
        while (
            current_ != end_ &&
            (*current_ == ' ' || *current_ == '\t' ||
             *current_ == '\r' || *current_ == '\n')
        ) {
            ++current_;
        }
    }

    bool consume(char expected) {
        skip_space();
        if (current_ != end_ && *current_ == expected) {
            ++current_;
            return true;
        }
        return false;
    }

    void expect(char expected) {
        if (!consume(expected)) {
            fail(std::string("expected '") + expected + "'");
        }
    }

    [[noreturn]] void fail(const std::string& message) const {
        throw std::runtime_error("invalid polygons JSON: " + message);
    }

    const char* current_;
    const char* end_;
};

class Sha256 {
public:
    static std::array<std::uint8_t, 32> digest(std::string_view input) {
        Sha256 state;
        state.update(
            reinterpret_cast<const std::uint8_t*>(input.data()),
            input.size()
        );
        return state.finish();
    }

private:
    static constexpr std::array<std::uint32_t, 64> constants_{
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
        0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
        0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
        0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
        0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
        0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
        0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
        0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
        0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
        0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
        0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
        0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
        0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
        0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
        0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
        0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
    };

    static std::uint32_t rotate_right(std::uint32_t value, int amount) {
        return (value >> amount) | (value << (32 - amount));
    }

    void update(const std::uint8_t* data, std::size_t size) {
        total_bytes_ += size;
        while (size > 0) {
            const std::size_t count =
                std::min(size, block_.size() - block_size_);
            std::copy_n(data, count, block_.data() + block_size_);
            data += count;
            size -= count;
            block_size_ += count;
            if (block_size_ == block_.size()) {
                transform(block_.data());
                block_size_ = 0;
            }
        }
    }

    std::array<std::uint8_t, 32> finish() {
        const std::uint64_t bit_count =
            static_cast<std::uint64_t>(total_bytes_) * 8U;
        block_[block_size_++] = 0x80U;
        if (block_size_ > 56) {
            std::fill(block_.begin() + block_size_, block_.end(), 0U);
            transform(block_.data());
            block_size_ = 0;
        }
        std::fill(block_.begin() + block_size_, block_.begin() + 56, 0U);
        for (int index = 0; index < 8; ++index) {
            block_[63 - index] =
                static_cast<std::uint8_t>(bit_count >> (index * 8));
        }
        transform(block_.data());

        std::array<std::uint8_t, 32> output{};
        for (std::size_t index = 0; index < state_.size(); ++index) {
            output[index * 4] =
                static_cast<std::uint8_t>(state_[index] >> 24);
            output[index * 4 + 1] =
                static_cast<std::uint8_t>(state_[index] >> 16);
            output[index * 4 + 2] =
                static_cast<std::uint8_t>(state_[index] >> 8);
            output[index * 4 + 3] =
                static_cast<std::uint8_t>(state_[index]);
        }
        return output;
    }

    void transform(const std::uint8_t* block) {
        std::array<std::uint32_t, 64> words{};
        for (int index = 0; index < 16; ++index) {
            const int offset = index * 4;
            words[index] =
                (static_cast<std::uint32_t>(block[offset]) << 24) |
                (static_cast<std::uint32_t>(block[offset + 1]) << 16) |
                (static_cast<std::uint32_t>(block[offset + 2]) << 8) |
                static_cast<std::uint32_t>(block[offset + 3]);
        }
        for (int index = 16; index < 64; ++index) {
            const std::uint32_t s0 =
                rotate_right(words[index - 15], 7) ^
                rotate_right(words[index - 15], 18) ^
                (words[index - 15] >> 3);
            const std::uint32_t s1 =
                rotate_right(words[index - 2], 17) ^
                rotate_right(words[index - 2], 19) ^
                (words[index - 2] >> 10);
            words[index] =
                words[index - 16] + s0 + words[index - 7] + s1;
        }

        auto [a, b, c, d, e, f, g, h] = state_;
        for (int index = 0; index < 64; ++index) {
            const std::uint32_t sum1 =
                rotate_right(e, 6) ^ rotate_right(e, 11) ^
                rotate_right(e, 25);
            const std::uint32_t choose = (e & f) ^ (~e & g);
            const std::uint32_t temp1 =
                h + sum1 + choose + constants_[index] + words[index];
            const std::uint32_t sum0 =
                rotate_right(a, 2) ^ rotate_right(a, 13) ^
                rotate_right(a, 22);
            const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t temp2 = sum0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<std::uint32_t, 8> state_{
        0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
        0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
    };
    std::array<std::uint8_t, 64> block_{};
    std::size_t block_size_{0};
    std::size_t total_bytes_{0};
};

std::array<std::uint8_t, 3> item_color(std::string_view color_key) {
    const auto digest = Sha256::digest(color_key);
    std::array<std::uint8_t, 3> bgr{};
    for (int index = 0; index < 3; ++index) {
        bgr[index] = static_cast<std::uint8_t>(
            96 + static_cast<int>(digest[index]) * 159 / 255
        );
    }
    return bgr;
}

class SqliteConnection {
public:
    explicit SqliteConnection(const fs::path& path) {
        const std::string uri =
            "file:" + path.string() + "?mode=ro&immutable=1";
        const int result = sqlite3_open_v2(
            uri.c_str(),
            &connection_,
            SQLITE_OPEN_READONLY | SQLITE_OPEN_URI,
            nullptr
        );
        if (result != SQLITE_OK) {
            const std::string message =
                connection_ ? sqlite3_errmsg(connection_) : "open failed";
            if (connection_) {
                sqlite3_close(connection_);
                connection_ = nullptr;
            }
            throw std::runtime_error("SQLite open failed: " + message);
        }
    }

    ~SqliteConnection() {
        if (connection_) {
            sqlite3_close(connection_);
        }
    }

    SqliteConnection(const SqliteConnection&) = delete;
    SqliteConnection& operator=(const SqliteConnection&) = delete;

    sqlite3* get() const { return connection_; }

private:
    sqlite3* connection_{nullptr};
};

class SqliteStatement {
public:
    SqliteStatement(sqlite3* connection, const char* sql) {
        const int result =
            sqlite3_prepare_v2(connection, sql, -1, &statement_, nullptr);
        if (result != SQLITE_OK) {
            throw std::runtime_error(
                std::string("SQLite prepare failed: ") +
                sqlite3_errmsg(connection)
            );
        }
    }

    ~SqliteStatement() {
        if (statement_) {
            sqlite3_finalize(statement_);
        }
    }

    sqlite3_stmt* get() const { return statement_; }

private:
    sqlite3_stmt* statement_{nullptr};
};

bool sqlite_table_exists(sqlite3* connection, std::string_view table) {
    SqliteStatement statement(
        connection,
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?1"
    );
    sqlite3_bind_text(
        statement.get(),
        1,
        table.data(),
        static_cast<int>(table.size()),
        SQLITE_TRANSIENT
    );
    return sqlite3_step(statement.get()) == SQLITE_ROW;
}

bool sqlite_column_exists(
    sqlite3* connection,
    std::string_view table,
    std::string_view column
) {
    const std::string query =
        "SELECT 1 FROM pragma_table_info(?1) WHERE name=?2";
    SqliteStatement statement(connection, query.c_str());
    sqlite3_bind_text(
        statement.get(),
        1,
        table.data(),
        static_cast<int>(table.size()),
        SQLITE_TRANSIENT
    );
    sqlite3_bind_text(
        statement.get(),
        2,
        column.data(),
        static_cast<int>(column.size()),
        SQLITE_TRANSIENT
    );
    return sqlite3_step(statement.get()) == SQLITE_ROW;
}

void bind_frame_range(
    sqlite3_stmt* statement,
    int start_frame,
    const std::optional<int>& end_frame
) {
    sqlite3_bind_int(statement, 1, start_frame);
    if (end_frame) {
        sqlite3_bind_int(statement, 2, *end_frame);
    } else {
        sqlite3_bind_null(statement, 2);
    }
}

std::string sqlite_text(sqlite3_stmt* statement, int column) {
    const auto* value = sqlite3_column_text(statement, column);
    return value
        ? std::string(reinterpret_cast<const char*>(value))
        : std::string();
}

void require_inference_role(
    sqlite3* connection,
    const fs::path& path,
    std::string_view role
) {
    SqliteStatement statement(
        connection,
        "SELECT COUNT(*) FROM model_executions WHERE role=?1"
    );
    sqlite3_bind_text(
        statement.get(),
        1,
        role.data(),
        static_cast<int>(role.size()),
        SQLITE_TRANSIENT
    );
    if (
        sqlite3_step(statement.get()) != SQLITE_ROW ||
        sqlite3_column_int64(statement.get(), 0) == 0
    ) {
        throw std::runtime_error(
            path.string() + ": required inference role is absent: " +
            std::string(role)
        );
    }
}

void validate_inference_schema(
    sqlite3* connection,
    const fs::path& path
) {
    if (!sqlite_table_exists(connection, "schema_info")) {
        throw std::runtime_error(
            path.string() + ": missing inference schema_info table"
        );
    }
    SqliteStatement statement(
        connection,
        "SELECT key, value FROM schema_info "
        "WHERE key IN ('schema_name', 'schema_version')"
    );
    std::string schema_name;
    std::string schema_version;
    while (sqlite3_step(statement.get()) == SQLITE_ROW) {
        const std::string key = sqlite_text(statement.get(), 0);
        if (key == "schema_name") {
            schema_name = sqlite_text(statement.get(), 1);
        } else if (key == "schema_version") {
            schema_version = sqlite_text(statement.get(), 1);
        }
    }
    if (
        schema_name != "instance-segmentation-unified-inference" ||
        (schema_version != "2" && schema_version != "3")
    ) {
        throw std::runtime_error(
            path.string() + ": unsupported inference schema " +
            schema_name + " version " + schema_version
        );
    }
}

FrameMasks load_postprocess_masks(
    const fs::path& path,
    int start_frame,
    const std::optional<int>& end_frame
) {
    SqliteConnection connection(path);
    const bool has_label = sqlite_column_exists(
        connection.get(),
        "masks",
        "label"
    );
    const std::string query =
        "SELECT frame, CAST(track_id AS TEXT), polygons, " +
        std::string(
            has_label ? "COALESCE(label, '')" : "''"
        ) +
        " FROM masks WHERE frame >= ?1 "
        "AND (?2 IS NULL OR frame <= ?2) ORDER BY frame, track_id";
    SqliteStatement statement(
        connection.get(),
        query.c_str()
    );
    bind_frame_range(statement.get(), start_frame, end_frame);

    FrameMasks frames;
    while (true) {
        const int result = sqlite3_step(statement.get());
        if (result == SQLITE_DONE) {
            break;
        }
        if (result != SQLITE_ROW) {
            throw std::runtime_error(
                std::string("SQLite read failed: ") +
                sqlite3_errmsg(connection.get())
            );
        }
        const int frame = sqlite3_column_int(statement.get(), 0);
        const std::string track_id = sqlite_text(statement.get(), 1);
        const std::string polygons_json = sqlite_text(statement.get(), 2);
        if (track_id.empty() || polygons_json.empty()) {
            throw std::runtime_error("mask row contains NULL track/polygons");
        }
        auto polygons = JsonPolygonParser(polygons_json).parse();
        if (!polygons.empty()) {
            frames[frame].push_back(
                Mask{
                    ItemKind::mask,
                    track_id,
                    sqlite_text(statement.get(), 3),
                    std::nullopt,
                    std::move(polygons),
                    std::nullopt,
                    item_color("track:" + track_id),
                }
            );
        }
    }
    return frames;
}

FrameMasks load_raw_masks(
    const fs::path& path,
    int start_frame,
    const std::optional<int>& end_frame
) {
    SqliteConnection connection(path);
    validate_inference_schema(connection.get(), path);
    require_inference_role(
        connection.get(),
        path,
        "instance_segmentation"
    );
    const bool has_classifications =
        sqlite_table_exists(connection.get(), "classifications") &&
        sqlite_column_exists(
            connection.get(),
            "classifications",
            "detection_id"
        ) &&
        sqlite_column_exists(
            connection.get(),
            "classifications",
            "class_name"
        ) &&
        sqlite_column_exists(
            connection.get(),
            "classifications",
            "score"
        );
    const std::string classification_columns = has_classifications
        ? "c.class_name, c.score"
        : "NULL, NULL";
    const std::string classification_join = has_classifications
        ? "LEFT JOIN classifications c ON c.detection_id=d.id "
        : "";
    const std::string query =
        "SELECT f.frame_index, d.id, d.class_name, d.score, " +
        classification_columns +
        ", sp.id, pt.x, pt.y "
        "FROM detections d "
        "JOIN frames f ON f.id=d.frame_id "
        "JOIN model_executions me ON me.id=d.model_execution_id "
        "JOIN segmentations s ON s.detection_id=d.id "
        "JOIN segmentation_polygons sp ON sp.detection_id=d.id "
        "JOIN segmentation_points pt ON pt.polygon_id=sp.id " +
        classification_join +
        "WHERE me.role='instance_segmentation' AND f.frame_index >= ?1 "
        "AND (?2 IS NULL OR f.frame_index <= ?2) "
        "ORDER BY f.frame_index, d.id, sp.polygon_index, pt.point_index";
    SqliteStatement statement(connection.get(), query.c_str());
    bind_frame_range(statement.get(), start_frame, end_frame);

    FrameMasks frames;
    std::int64_t current_detection = -1;
    std::int64_t current_polygon = -1;
    int frame = 0;
    std::string label;
    double score = 0.0;
    std::vector<Polygon> polygons;
    Polygon polygon;
    auto finish_polygon = [&]() {
        if (polygon.size() >= 3) {
            polygons.push_back(std::move(polygon));
        }
        polygon.clear();
    };
    auto finish_detection = [&]() {
        if (current_detection < 0) {
            return;
        }
        finish_polygon();
        if (!polygons.empty()) {
            frames[frame].push_back(
                Mask{
                    ItemKind::mask,
                    "",
                    label,
                    score,
                    std::move(polygons),
                    std::nullopt,
                    item_color("raw:" + label),
                }
            );
        }
        polygons.clear();
    };

    while (true) {
        const int result = sqlite3_step(statement.get());
        if (result == SQLITE_DONE) {
            break;
        }
        if (result != SQLITE_ROW) {
            throw std::runtime_error(
                std::string("raw inference SQLite read failed: ") +
                sqlite3_errmsg(connection.get())
            );
        }
        const std::int64_t detection =
            sqlite3_column_int64(statement.get(), 1);
        const std::int64_t polygon_id =
            sqlite3_column_int64(statement.get(), 6);
        if (detection != current_detection) {
            finish_detection();
            current_detection = detection;
            current_polygon = polygon_id;
            frame = sqlite3_column_int(statement.get(), 0);
            const std::string detector_label = sqlite_text(
                statement.get(),
                2
            );
            const std::string classified_label = sqlite_text(
                statement.get(),
                4
            );
            label = classified_label.empty()
                ? detector_label
                : classified_label;
            score = sqlite3_column_type(statement.get(), 5) == SQLITE_NULL
                ? sqlite3_column_double(statement.get(), 3)
                : sqlite3_column_double(statement.get(), 5);
        } else if (polygon_id != current_polygon) {
            finish_polygon();
            current_polygon = polygon_id;
        }
        polygon.push_back(
            Point{
                sqlite3_column_double(statement.get(), 7),
                sqlite3_column_double(statement.get(), 8),
            }
        );
    }
    finish_detection();
    return frames;
}

FrameMasks load_faces(
    const fs::path& path,
    int start_frame,
    const std::optional<int>& end_frame
) {
    SqliteConnection connection(path);
    validate_inference_schema(connection.get(), path);
    require_inference_role(connection.get(), path, "face_detection");
    SqliteStatement statement(
        connection.get(),
        "SELECT f.frame_index, d.class_name, d.score, "
        "d.x1, d.y1, d.x2, d.y2 "
        "FROM detections d "
        "JOIN frames f ON f.id=d.frame_id "
        "JOIN model_executions me ON me.id=d.model_execution_id "
        "WHERE me.role='face_detection' AND f.frame_index >= ?1 "
        "AND (?2 IS NULL OR f.frame_index <= ?2) "
        "ORDER BY f.frame_index, d.id"
    );
    bind_frame_range(statement.get(), start_frame, end_frame);
    FrameMasks frames;
    while (true) {
        const int result = sqlite3_step(statement.get());
        if (result == SQLITE_DONE) {
            break;
        }
        if (result != SQLITE_ROW) {
            throw std::runtime_error(
                std::string("face inference SQLite read failed: ") +
                sqlite3_errmsg(connection.get())
            );
        }
        const int frame = sqlite3_column_int(statement.get(), 0);
        const std::string label = sqlite_text(statement.get(), 1);
        frames[frame].push_back(
            Mask{
                ItemKind::face,
                "",
                label,
                sqlite3_column_double(statement.get(), 2),
                {},
                Box{
                    sqlite3_column_double(statement.get(), 3),
                    sqlite3_column_double(statement.get(), 4),
                    sqlite3_column_double(statement.get(), 5),
                    sqlite3_column_double(statement.get(), 6),
                },
                item_color("face:" + label),
            }
        );
    }
    return frames;
}

void merge_frame_items(FrameMasks& destination, FrameMasks source) {
    for (auto& [frame, items] : source) {
        auto& target = destination[frame];
        target.insert(
            target.end(),
            std::make_move_iterator(items.begin()),
            std::make_move_iterator(items.end())
        );
    }
}

void validate_sqlite_video_metadata(
    const fs::path& path,
    int width,
    int height,
    double fps
) {
    SqliteConnection connection(path);
    if (!sqlite_table_exists(connection.get(), "videos")) {
        return;
    }
    SqliteStatement statement(
        connection.get(),
        "SELECT width, height, fps FROM videos ORDER BY id LIMIT 1"
    );
    if (sqlite3_step(statement.get()) != SQLITE_ROW) {
        return;
    }
    if (
        sqlite3_column_type(statement.get(), 0) != SQLITE_NULL &&
        sqlite3_column_int(statement.get(), 0) != width
    ) {
        throw std::runtime_error(
            path.string() + ": SQLite/video width mismatch"
        );
    }
    if (
        sqlite3_column_type(statement.get(), 1) != SQLITE_NULL &&
        sqlite3_column_int(statement.get(), 1) != height
    ) {
        throw std::runtime_error(
            path.string() + ": SQLite/video height mismatch"
        );
    }
    if (
        sqlite3_column_type(statement.get(), 2) != SQLITE_NULL &&
        std::abs(sqlite3_column_double(statement.get(), 2) - fps) > 0.02
    ) {
        throw std::runtime_error(
            path.string() + ": SQLite/video FPS mismatch"
        );
    }
}

std::uint8_t clamp_byte(double value) {
    return static_cast<std::uint8_t>(
        std::clamp(std::lround(value), 0L, 255L)
    );
}

struct YuvColor {
    std::uint8_t y;
    std::uint8_t u;
    std::uint8_t v;
};

YuvColor bgr_to_bt709_limited(const std::array<std::uint8_t, 3>& bgr) {
    const double b = bgr[0];
    const double g = bgr[1];
    const double r = bgr[2];
    return YuvColor{
        clamp_byte(16.0 + 0.182586 * r + 0.614231 * g + 0.062007 * b),
        clamp_byte(128.0 - 0.100644 * r - 0.338572 * g + 0.439216 * b),
        clamp_byte(128.0 + 0.439216 * r - 0.398942 * g - 0.040274 * b),
    };
}

std::uint8_t blend(std::uint8_t background, std::uint8_t foreground, double a) {
    return clamp_byte(
        a * static_cast<double>(foreground) +
        (1.0 - a) * static_cast<double>(background)
    );
}

std::uint8_t blend_fixed(
    std::uint8_t background,
    std::uint8_t foreground,
    int alpha
) {
    return static_cast<std::uint8_t>(
        (
            static_cast<int>(foreground) * alpha +
            static_cast<int>(background) * (255 - alpha) +
            127
        ) /
        255
    );
}

void blend_span(
    AVFrame* frame,
    int y,
    int first_x,
    int last_x,
    const YuvColor& color,
    int alpha
) {
    auto* luma = frame->data[0] + y * frame->linesize[0];
    for (int x = first_x; x <= last_x; ++x) {
        luma[x] = blend_fixed(luma[x], color.y, alpha);
    }
    if ((y & 1) != 0) {
        return;
    }
    const int first_chroma_x = (first_x + 1) / 2;
    const int last_chroma_x = last_x / 2;
    auto* u = frame->data[1] + (y / 2) * frame->linesize[1];
    if (frame->format == AV_PIX_FMT_NV12) {
        for (
            int chroma_x = first_chroma_x;
            chroma_x <= last_chroma_x;
            ++chroma_x
        ) {
            auto& u_value = u[chroma_x * 2];
            auto& v_value = u[chroma_x * 2 + 1];
            u_value = blend_fixed(u_value, color.u, alpha);
            v_value = blend_fixed(v_value, color.v, alpha);
        }
    } else {
        auto* v = frame->data[2] + (y / 2) * frame->linesize[2];
        for (
            int chroma_x = first_chroma_x;
            chroma_x <= last_chroma_x;
            ++chroma_x
        ) {
            u[chroma_x] = blend_fixed(u[chroma_x], color.u, alpha);
            v[chroma_x] = blend_fixed(v[chroma_x], color.v, alpha);
        }
    }
}

void blend_pixel(
    AVFrame* frame,
    int x,
    int y,
    const YuvColor& color,
    double alpha,
    bool write_chroma
) {
    if (x < 0 || y < 0 || x >= frame->width || y >= frame->height) {
        return;
    }
    auto& luma = frame->data[0][y * frame->linesize[0] + x];
    luma = blend(luma, color.y, alpha);
    if (write_chroma) {
        const int chroma_x = x / 2;
        const int chroma_y = y / 2;
        if (frame->format == AV_PIX_FMT_NV12) {
            auto& u = frame->data[1][
                chroma_y * frame->linesize[1] + chroma_x * 2
            ];
            auto& v = frame->data[1][
                chroma_y * frame->linesize[1] + chroma_x * 2 + 1
            ];
            u = blend(u, color.u, alpha);
            v = blend(v, color.v, alpha);
        } else {
            auto& u =
                frame->data[1][chroma_y * frame->linesize[1] + chroma_x];
            auto& v =
                frame->data[2][chroma_y * frame->linesize[2] + chroma_x];
            u = blend(u, color.u, alpha);
            v = blend(v, color.v, alpha);
        }
    }
}

void fill_polygon(
    AVFrame* frame,
    const Polygon& polygon,
    const YuvColor& color,
    double alpha
) {
    if (polygon.size() < 3 || alpha <= 0.0) {
        return;
    }
    const int fixed_alpha = std::clamp(
        static_cast<int>(std::lround(alpha * 255.0)),
        0,
        255
    );
    double minimum_y = polygon.front().y;
    double maximum_y = polygon.front().y;
    for (const auto& point : polygon) {
        minimum_y = std::min(minimum_y, point.y);
        maximum_y = std::max(maximum_y, point.y);
    }
    const int first_y = std::max(0, static_cast<int>(std::ceil(minimum_y)));
    const int last_y = std::min(
        frame->height - 1,
        static_cast<int>(std::floor(maximum_y))
    );
    std::vector<double> intersections;
    intersections.reserve(polygon.size());

    for (int y = first_y; y <= last_y; ++y) {
        const double scan_y = static_cast<double>(y) + 0.5;
        intersections.clear();
        for (std::size_t index = 0; index < polygon.size(); ++index) {
            const Point& first = polygon[index];
            const Point& second = polygon[(index + 1) % polygon.size()];
            const bool crosses =
                (first.y <= scan_y && second.y > scan_y) ||
                (second.y <= scan_y && first.y > scan_y);
            if (!crosses) {
                continue;
            }
            const double fraction =
                (scan_y - first.y) / (second.y - first.y);
            intersections.push_back(
                first.x + fraction * (second.x - first.x)
            );
        }
        std::sort(intersections.begin(), intersections.end());
        for (
            std::size_t pair = 0;
            pair + 1 < intersections.size();
            pair += 2
        ) {
            const int first_x = std::max(
                0,
                static_cast<int>(std::ceil(intersections[pair]))
            );
            const int last_x = std::min(
                frame->width - 1,
                static_cast<int>(std::floor(intersections[pair + 1]))
            );
            if (first_x <= last_x) {
                blend_span(
                    frame,
                    y,
                    first_x,
                    last_x,
                    color,
                    fixed_alpha
                );
            }
        }
    }
}

void draw_disc(
    AVFrame* frame,
    int center_x,
    int center_y,
    int radius,
    const YuvColor& color
) {
    for (int y = center_y - radius; y <= center_y + radius; ++y) {
        for (int x = center_x - radius; x <= center_x + radius; ++x) {
            const int dx = x - center_x;
            const int dy = y - center_y;
            if (dx * dx + dy * dy <= radius * radius) {
                blend_pixel(frame, x, y, color, 1.0, true);
            }
        }
    }
}

void draw_line(
    AVFrame* frame,
    Point first,
    Point second,
    int thickness,
    const YuvColor& color
) {
    int x0 = std::clamp(
        static_cast<int>(std::lround(first.x)),
        0,
        frame->width - 1
    );
    int y0 = std::clamp(
        static_cast<int>(std::lround(first.y)),
        0,
        frame->height - 1
    );
    const int x1 = std::clamp(
        static_cast<int>(std::lround(second.x)),
        0,
        frame->width - 1
    );
    const int y1 = std::clamp(
        static_cast<int>(std::lround(second.y)),
        0,
        frame->height - 1
    );
    const int dx = std::abs(x1 - x0);
    const int sx = x0 < x1 ? 1 : -1;
    const int dy = -std::abs(y1 - y0);
    const int sy = y0 < y1 ? 1 : -1;
    int error = dx + dy;
    const int radius = std::max(0, thickness / 2);

    while (true) {
        draw_disc(frame, x0, y0, radius, color);
        if (x0 == x1 && y0 == y1) {
            break;
        }
        const int doubled = error * 2;
        if (doubled >= dy) {
            error += dy;
            x0 += sx;
        }
        if (doubled <= dx) {
            error += dx;
            y0 += sy;
        }
    }
}

void draw_cpu_label(AVFrame* frame, const Mask& item);

void draw_masks(
    AVFrame* frame,
    const std::vector<Mask>& masks,
    double alpha,
    int outline_thickness,
    int box_thickness,
    bool show_labels
) {
    for (const auto& mask : masks) {
        if (mask.kind != ItemKind::mask) {
            continue;
        }
        const YuvColor color = bgr_to_bt709_limited(mask.bgr);
        for (const auto& polygon : mask.polygons) {
            fill_polygon(frame, polygon, color, alpha);
        }
    }
    if (outline_thickness > 0) {
        for (const auto& mask : masks) {
            if (mask.kind != ItemKind::mask) {
                continue;
            }
            const YuvColor color = bgr_to_bt709_limited(mask.bgr);
            for (const auto& polygon : mask.polygons) {
                if (polygon.size() < 2) {
                    continue;
                }
                for (
                    std::size_t index = 0;
                    index < polygon.size();
                    ++index
                ) {
                    draw_line(
                        frame,
                        polygon[index],
                        polygon[(index + 1) % polygon.size()],
                        outline_thickness,
                        color
                    );
                }
            }
        }
    }
    if (show_labels) {
        for (const auto& item : masks) {
            if (item.kind == ItemKind::mask) {
                draw_cpu_label(frame, item);
            }
        }
    }
    if (box_thickness > 0) {
        for (const auto& item : masks) {
            if (item.kind != ItemKind::face || !item.box) {
                continue;
            }
            const YuvColor color = bgr_to_bt709_limited(item.bgr);
            const Box& box = *item.box;
            const Point top_left{box.x1, box.y1};
            const Point top_right{box.x2, box.y1};
            const Point bottom_right{box.x2, box.y2};
            const Point bottom_left{box.x1, box.y2};
            draw_line(
                frame,
                top_left,
                top_right,
                box_thickness,
                color
            );
            draw_line(
                frame,
                top_right,
                bottom_right,
                box_thickness,
                color
            );
            draw_line(
                frame,
                bottom_right,
                bottom_left,
                box_thickness,
                color
            );
            draw_line(
                frame,
                bottom_left,
                top_left,
                box_thickness,
                color
            );
        }
    }
    if (show_labels) {
        for (const auto& item : masks) {
            if (item.kind == ItemKind::face) {
                draw_cpu_label(frame, item);
            }
        }
    }
}

void append_cuda_fill_spans(
    std::vector<CudaOverlaySpan>& output,
    const Polygon& polygon,
    const YuvColor& color,
    int alpha,
    int width,
    int height
) {
    if (polygon.size() < 3 || alpha <= 0) {
        return;
    }
    double minimum_y = polygon.front().y;
    double maximum_y = polygon.front().y;
    for (const auto& point : polygon) {
        minimum_y = std::min(minimum_y, point.y);
        maximum_y = std::max(maximum_y, point.y);
    }
    const int first_y = std::max(0, static_cast<int>(std::ceil(minimum_y)));
    const int last_y = std::min(
        height - 1,
        static_cast<int>(std::floor(maximum_y))
    );
    std::vector<double> intersections;
    intersections.reserve(polygon.size());
    for (int y = first_y; y <= last_y; ++y) {
        const double scan_y = static_cast<double>(y) + 0.5;
        intersections.clear();
        for (std::size_t index = 0; index < polygon.size(); ++index) {
            const Point& first = polygon[index];
            const Point& second = polygon[(index + 1) % polygon.size()];
            const bool crosses =
                (first.y <= scan_y && second.y > scan_y) ||
                (second.y <= scan_y && first.y > scan_y);
            if (!crosses) {
                continue;
            }
            const double fraction =
                (scan_y - first.y) / (second.y - first.y);
            intersections.push_back(
                first.x + fraction * (second.x - first.x)
            );
        }
        std::sort(intersections.begin(), intersections.end());
        for (
            std::size_t pair = 0;
            pair + 1 < intersections.size();
            pair += 2
        ) {
            const int first_x = std::max(
                0,
                static_cast<int>(std::ceil(intersections[pair]))
            );
            const int last_x = std::min(
                width - 1,
                static_cast<int>(std::floor(intersections[pair + 1]))
            );
            if (first_x <= last_x) {
                output.push_back(
                    CudaOverlaySpan{
                        y,
                        first_x,
                        last_x,
                        color.y,
                        color.u,
                        color.v,
                        static_cast<std::uint8_t>(alpha),
                    }
                );
            }
        }
    }
}

void append_cuda_disc_spans(
    std::vector<CudaOverlaySpan>& output,
    int center_x,
    int center_y,
    int radius,
    const YuvColor& color,
    int width,
    int height
) {
    for (int offset_y = -radius; offset_y <= radius; ++offset_y) {
        const int y = center_y + offset_y;
        if (y < 0 || y >= height) {
            continue;
        }
        const int half_width = static_cast<int>(
            std::floor(
                std::sqrt(
                    static_cast<double>(
                        radius * radius - offset_y * offset_y
                    )
                )
            )
        );
        const int first_x = std::max(0, center_x - half_width);
        const int last_x = std::min(width - 1, center_x + half_width);
        if (first_x <= last_x) {
            output.push_back(
                CudaOverlaySpan{
                    y,
                    first_x,
                    last_x,
                    color.y,
                    color.u,
                    color.v,
                    255,
                }
            );
        }
    }
}

void append_cuda_line_spans(
    std::vector<CudaOverlaySpan>& output,
    Point first,
    Point second,
    int thickness,
    const YuvColor& color,
    int width,
    int height
) {
    int x0 = std::clamp(
        static_cast<int>(std::lround(first.x)),
        0,
        width - 1
    );
    int y0 = std::clamp(
        static_cast<int>(std::lround(first.y)),
        0,
        height - 1
    );
    const int x1 = std::clamp(
        static_cast<int>(std::lround(second.x)),
        0,
        width - 1
    );
    const int y1 = std::clamp(
        static_cast<int>(std::lround(second.y)),
        0,
        height - 1
    );
    const int dx = std::abs(x1 - x0);
    const int sx = x0 < x1 ? 1 : -1;
    const int dy = -std::abs(y1 - y0);
    const int sy = y0 < y1 ? 1 : -1;
    int error = dx + dy;
    const int radius = std::max(0, thickness / 2);
    auto append_span = [&](int y, int first_x, int last_x) {
        if (y < 0 || y >= height) {
            return;
        }
        first_x = std::clamp(first_x, 0, width - 1);
        last_x = std::clamp(last_x, 0, width - 1);
        if (first_x <= last_x) {
            output.push_back(
                CudaOverlaySpan{
                    y,
                    first_x,
                    last_x,
                    color.y,
                    color.u,
                    color.v,
                    255,
                }
            );
        }
    };
    if (y0 == y1) {
        const int first_x = std::min(x0, x1);
        const int last_x = std::max(x0, x1);
        for (int offset_y = -radius; offset_y <= radius; ++offset_y) {
            const int half_width = static_cast<int>(
                std::floor(
                    std::sqrt(
                        static_cast<double>(
                            radius * radius - offset_y * offset_y
                        )
                    )
                )
            );
            append_span(
                y0 + offset_y,
                first_x - half_width,
                last_x + half_width
            );
        }
        return;
    }
    if (x0 == x1) {
        const int first_y = std::min(y0, y1);
        const int last_y = std::max(y0, y1);
        for (
            int y = first_y - radius;
            y <= last_y + radius;
            ++y
        ) {
            int vertical_offset = 0;
            if (y < first_y) {
                vertical_offset = first_y - y;
            } else if (y > last_y) {
                vertical_offset = y - last_y;
            }
            const int half_width = static_cast<int>(
                std::floor(
                    std::sqrt(
                        static_cast<double>(
                            radius * radius -
                            vertical_offset * vertical_offset
                        )
                    )
                )
            );
            append_span(y, x0 - half_width, x0 + half_width);
        }
        return;
    }
    while (true) {
        append_cuda_disc_spans(
            output,
            x0,
            y0,
            radius,
            color,
            width,
            height
        );
        if (x0 == x1 && y0 == y1) {
            break;
        }
        const int doubled = error * 2;
        if (doubled >= dy) {
            error += dy;
            x0 += sx;
        }
        if (doubled <= dx) {
            error += dx;
            y0 += sy;
        }
    }
}

std::string overlay_label(const Mask& item) {
    std::vector<std::string> components;
    if (!item.track_id.empty()) {
        components.push_back("T" + item.track_id);
    }
    const bool ascii_label = std::all_of(
        item.label.begin(),
        item.label.end(),
        [](unsigned char character) { return character < 128; }
    );
    if (!item.label.empty() && ascii_label) {
        std::string upper = item.label;
        std::transform(
            upper.begin(),
            upper.end(),
            upper.begin(),
            [](unsigned char character) {
                return static_cast<char>(std::toupper(character));
            }
        );
        components.push_back(std::move(upper));
    }
    if (item.score) {
        std::ostringstream score;
        score << std::fixed << std::setprecision(2) << *item.score;
        components.push_back(score.str());
    }
    std::ostringstream output;
    for (std::size_t index = 0; index < components.size(); ++index) {
        if (index) {
            output << ' ';
        }
        output << components[index];
    }
    return output.str();
}

Point label_anchor(const Mask& item) {
    if (item.kind == ItemKind::face && item.box) {
        return Point{item.box->x1, item.box->y1 - 3.0};
    }
    double minimum_x = std::numeric_limits<double>::max();
    double minimum_y = std::numeric_limits<double>::max();
    for (const auto& polygon : item.polygons) {
        for (const auto& point : polygon) {
            minimum_x = std::min(minimum_x, point.x);
            minimum_y = std::min(minimum_y, point.y);
        }
    }
    if (!std::isfinite(minimum_x) || !std::isfinite(minimum_y)) {
        return Point{};
    }
    return Point{minimum_x, minimum_y - 3.0};
}

enum SevenSegment : std::uint8_t {
    segment_top = 1U << 0,
    segment_upper_right = 1U << 1,
    segment_lower_right = 1U << 2,
    segment_bottom = 1U << 3,
    segment_lower_left = 1U << 4,
    segment_upper_left = 1U << 5,
    segment_middle = 1U << 6,
};

std::uint8_t glyph_segments(char character) {
    constexpr std::uint8_t all =
        segment_top | segment_upper_right | segment_lower_right |
        segment_bottom | segment_lower_left | segment_upper_left |
        segment_middle;
    switch (character) {
    case '0': return all & ~segment_middle;
    case '1': return segment_upper_right | segment_lower_right;
    case '2':
        return segment_top | segment_upper_right | segment_middle |
            segment_lower_left | segment_bottom;
    case '3':
        return segment_top | segment_upper_right | segment_middle |
            segment_lower_right | segment_bottom;
    case '4':
        return segment_upper_left | segment_middle |
            segment_upper_right | segment_lower_right;
    case '5':
        return segment_top | segment_upper_left | segment_middle |
            segment_lower_right | segment_bottom;
    case '6':
        return segment_top | segment_upper_left | segment_middle |
            segment_lower_left | segment_lower_right | segment_bottom;
    case '7':
        return segment_top | segment_upper_right | segment_lower_right;
    case '8': return all;
    case '9':
        return segment_top | segment_upper_left | segment_upper_right |
            segment_middle | segment_lower_right | segment_bottom;
    case 'A':
        return segment_top | segment_upper_left | segment_upper_right |
            segment_middle | segment_lower_left | segment_lower_right;
    case 'B':
        return segment_upper_left | segment_lower_left | segment_middle |
            segment_lower_right | segment_bottom;
    case 'C':
        return segment_top | segment_upper_left | segment_lower_left |
            segment_bottom;
    case 'D':
        return segment_upper_right | segment_lower_right | segment_middle |
            segment_lower_left | segment_bottom;
    case 'E':
        return segment_top | segment_upper_left | segment_middle |
            segment_lower_left | segment_bottom;
    case 'F':
        return segment_top | segment_upper_left | segment_middle |
            segment_lower_left;
    case 'G':
        return segment_top | segment_upper_left | segment_lower_left |
            segment_bottom | segment_lower_right | segment_middle;
    case 'H':
        return segment_upper_left | segment_lower_left | segment_middle |
            segment_upper_right | segment_lower_right;
    case 'I':
        return segment_upper_right | segment_lower_right;
    case 'J':
        return segment_upper_right | segment_lower_right |
            segment_lower_left | segment_bottom;
    case 'K':
    case 'X':
        return segment_upper_left | segment_lower_left | segment_middle |
            segment_upper_right | segment_lower_right;
    case 'L':
        return segment_upper_left | segment_lower_left | segment_bottom;
    case 'M':
    case 'N':
        return segment_upper_left | segment_lower_left |
            segment_upper_right | segment_lower_right | segment_top;
    case 'O': return all & ~segment_middle;
    case 'P':
        return segment_top | segment_upper_left | segment_upper_right |
            segment_middle | segment_lower_left;
    case 'Q':
        return all & ~segment_middle;
    case 'R':
        return segment_top | segment_upper_left | segment_upper_right |
            segment_middle | segment_lower_left | segment_lower_right;
    case 'S':
        return segment_top | segment_upper_left | segment_middle |
            segment_lower_right | segment_bottom;
    case 'T':
        return segment_upper_left | segment_lower_left |
            segment_middle | segment_bottom;
    case 'U':
        return segment_upper_left | segment_lower_left |
            segment_upper_right | segment_lower_right | segment_bottom;
    case 'V':
        return segment_upper_left | segment_lower_left |
            segment_upper_right | segment_lower_right | segment_bottom;
    case 'W':
        return segment_upper_left | segment_lower_left |
            segment_upper_right | segment_lower_right | segment_bottom;
    case 'Y':
        return segment_upper_left | segment_upper_right |
            segment_middle | segment_lower_right | segment_bottom;
    case 'Z':
        return segment_top | segment_upper_right | segment_middle |
            segment_lower_left | segment_bottom;
    case '-': return segment_middle;
    case '_': return segment_bottom;
    default: return 0;
    }
}

constexpr std::array<std::pair<Point, Point>, 7> glyph_lines{
    std::pair{Point{1, 0}, Point{5, 0}},
    std::pair{Point{6, 1}, Point{6, 4}},
    std::pair{Point{6, 6}, Point{6, 9}},
    std::pair{Point{1, 10}, Point{5, 10}},
    std::pair{Point{0, 6}, Point{0, 9}},
    std::pair{Point{0, 1}, Point{0, 4}},
    std::pair{Point{1, 5}, Point{5, 5}},
};

struct LabelLayout {
    std::string text;
    int x{};
    int y{};
    int width{};
    static constexpr int height = 14;
};

LabelLayout make_label_layout(
    const Mask& item,
    int frame_width,
    int frame_height
) {
    LabelLayout layout;
    layout.text = overlay_label(item);
    if (layout.text.size() > 80) {
        layout.text.resize(80);
    }
    layout.width = std::min(
        frame_width,
        static_cast<int>(layout.text.size()) * 8 + 4
    );
    const Point anchor = label_anchor(item);
    layout.x = std::clamp(
        static_cast<int>(std::lround(anchor.x)),
        0,
        std::max(0, frame_width - layout.width)
    );
    layout.y = std::clamp(
        static_cast<int>(std::lround(anchor.y)) - LabelLayout::height,
        0,
        std::max(0, frame_height - LabelLayout::height)
    );
    return layout;
}

void append_cuda_solid_rectangle(
    std::vector<CudaOverlaySpan>& output,
    int first_x,
    int first_y,
    int last_x,
    int last_y,
    const YuvColor& color,
    int width,
    int height
) {
    first_x = std::clamp(first_x, 0, width - 1);
    last_x = std::clamp(last_x, 0, width - 1);
    first_y = std::clamp(first_y, 0, height - 1);
    last_y = std::clamp(last_y, 0, height - 1);
    if (first_x > last_x || first_y > last_y) {
        return;
    }
    for (int y = first_y; y <= last_y; ++y) {
        output.push_back(
            CudaOverlaySpan{
                y,
                first_x,
                last_x,
                color.y,
                color.u,
                color.v,
                255,
            }
        );
    }
}

void append_cuda_label(
    std::vector<CudaOverlaySpan>& output,
    std::vector<std::size_t>& batch_ends,
    const Mask& item,
    int width,
    int height
) {
    const LabelLayout layout = make_label_layout(item, width, height);
    if (layout.text.empty() || layout.width <= 0) {
        return;
    }
    const YuvColor background = bgr_to_bt709_limited({18, 18, 18});
    append_cuda_solid_rectangle(
        output,
        layout.x,
        layout.y,
        layout.x + layout.width - 1,
        layout.y + LabelLayout::height - 1,
        background,
        width,
        height
    );
    batch_ends.push_back(output.size());

    const YuvColor foreground = bgr_to_bt709_limited(item.bgr);
    const std::size_t text_start = output.size();
    for (std::size_t index = 0; index < layout.text.size(); ++index) {
        const char character = layout.text[index];
        const int origin_x = layout.x + 2 + static_cast<int>(index) * 8;
        if (character == '.') {
            append_cuda_disc_spans(
                output,
                origin_x + 3,
                layout.y + 11,
                1,
                foreground,
                width,
                height
            );
            continue;
        }
        if (character == ':') {
            append_cuda_disc_spans(
                output,
                origin_x + 3,
                layout.y + 4,
                1,
                foreground,
                width,
                height
            );
            append_cuda_disc_spans(
                output,
                origin_x + 3,
                layout.y + 9,
                1,
                foreground,
                width,
                height
            );
            continue;
        }
        const std::uint8_t segments = glyph_segments(character);
        for (std::size_t segment = 0; segment < glyph_lines.size(); ++segment) {
            if ((segments & (1U << segment)) == 0) {
                continue;
            }
            append_cuda_line_spans(
                output,
                Point{
                    origin_x + glyph_lines[segment].first.x,
                    layout.y + 1 + glyph_lines[segment].first.y,
                },
                Point{
                    origin_x + glyph_lines[segment].second.x,
                    layout.y + 1 + glyph_lines[segment].second.y,
                },
                1,
                foreground,
                width,
                height
            );
        }
    }
    if (output.size() != text_start) {
        batch_ends.push_back(output.size());
    }
}

void draw_cpu_label(AVFrame* frame, const Mask& item) {
    const LabelLayout layout = make_label_layout(
        item,
        frame->width,
        frame->height
    );
    if (layout.text.empty() || layout.width <= 0) {
        return;
    }
    const YuvColor background = bgr_to_bt709_limited({18, 18, 18});
    for (
        int y = layout.y;
        y < layout.y + LabelLayout::height;
        ++y
    ) {
        blend_span(
            frame,
            y,
            layout.x,
            layout.x + layout.width - 1,
            background,
            255
        );
    }
    const YuvColor foreground = bgr_to_bt709_limited(item.bgr);
    for (std::size_t index = 0; index < layout.text.size(); ++index) {
        const char character = layout.text[index];
        const int origin_x = layout.x + 2 + static_cast<int>(index) * 8;
        if (character == '.') {
            draw_disc(frame, origin_x + 3, layout.y + 11, 1, foreground);
            continue;
        }
        if (character == ':') {
            draw_disc(frame, origin_x + 3, layout.y + 4, 1, foreground);
            draw_disc(frame, origin_x + 3, layout.y + 9, 1, foreground);
            continue;
        }
        const std::uint8_t segments = glyph_segments(character);
        for (std::size_t segment = 0; segment < glyph_lines.size(); ++segment) {
            if ((segments & (1U << segment)) == 0) {
                continue;
            }
            draw_line(
                frame,
                Point{
                    origin_x + glyph_lines[segment].first.x,
                    layout.y + 1 + glyph_lines[segment].first.y,
                },
                Point{
                    origin_x + glyph_lines[segment].second.x,
                    layout.y + 1 + glyph_lines[segment].second.y,
                },
                1,
                foreground
            );
        }
    }
}

class CudaOverlayContext {
public:
    explicit CudaOverlayContext(int device_index) {
        std::array<char, 512> error{};
        context_ = cuda_overlay_create(
            device_index,
            error.data(),
            error.size()
        );
        if (!context_) {
            throw std::runtime_error(
                std::string("CUDA overlay initialization failed: ") +
                error.data()
            );
        }
    }

    ~CudaOverlayContext() { cuda_overlay_destroy(context_); }

    CudaOverlayContext(const CudaOverlayContext&) = delete;
    CudaOverlayContext& operator=(const CudaOverlayContext&) = delete;

    void apply(
        AVFrame* frame,
        const std::vector<Mask>& masks,
        double alpha,
        int outline_thickness,
        int box_thickness,
        bool show_labels
    ) {
        fill_spans_.clear();
        fill_batch_ends_.clear();
        outline_spans_.clear();
        outline_batch_ends_.clear();
        const int fixed_alpha = std::clamp(
            static_cast<int>(std::lround(alpha * 255.0)),
            0,
            255
        );
        for (const auto& mask : masks) {
            if (mask.kind != ItemKind::mask) {
                continue;
            }
            const YuvColor color = bgr_to_bt709_limited(mask.bgr);
            for (const auto& polygon : mask.polygons) {
                const std::size_t batch_start = fill_spans_.size();
                append_cuda_fill_spans(
                    fill_spans_,
                    polygon,
                    color,
                    fixed_alpha,
                    frame->width,
                    frame->height
                );
                if (fill_spans_.size() != batch_start) {
                    fill_batch_ends_.push_back(fill_spans_.size());
                }
            }
        }
        if (outline_thickness > 0) {
            for (const auto& mask : masks) {
                if (mask.kind != ItemKind::mask) {
                    continue;
                }
                const std::size_t batch_start = outline_spans_.size();
                const YuvColor color = bgr_to_bt709_limited(mask.bgr);
                for (const auto& polygon : mask.polygons) {
                    if (polygon.size() < 2) {
                        continue;
                    }
                    for (
                        std::size_t index = 0;
                        index < polygon.size();
                        ++index
                    ) {
                        append_cuda_line_spans(
                            outline_spans_,
                            polygon[index],
                            polygon[(index + 1) % polygon.size()],
                            outline_thickness,
                            color,
                            frame->width,
                            frame->height
                        );
                    }
                }
                if (outline_spans_.size() != batch_start) {
                    outline_batch_ends_.push_back(outline_spans_.size());
                }
            }
        }
        if (show_labels) {
            for (const auto& item : masks) {
                if (item.kind == ItemKind::mask) {
                    append_cuda_label(
                        outline_spans_,
                        outline_batch_ends_,
                        item,
                        frame->width,
                        frame->height
                    );
                }
            }
        }
        if (box_thickness > 0) {
            for (const auto& item : masks) {
                if (item.kind != ItemKind::face || !item.box) {
                    continue;
                }
                const std::size_t batch_start = outline_spans_.size();
                const YuvColor color = bgr_to_bt709_limited(item.bgr);
                const Box& box = *item.box;
                const std::array<Point, 4> points{
                    Point{box.x1, box.y1},
                    Point{box.x2, box.y1},
                    Point{box.x2, box.y2},
                    Point{box.x1, box.y2},
                };
                for (std::size_t index = 0; index < points.size(); ++index) {
                    append_cuda_line_spans(
                        outline_spans_,
                        points[index],
                        points[(index + 1) % points.size()],
                        box_thickness,
                        color,
                        frame->width,
                        frame->height
                    );
                }
                if (outline_spans_.size() != batch_start) {
                    outline_batch_ends_.push_back(outline_spans_.size());
                }
                if (show_labels) {
                    append_cuda_label(
                        outline_spans_,
                        outline_batch_ends_,
                        item,
                        frame->width,
                        frame->height
                    );
                }
            }
        } else if (show_labels) {
            for (const auto& item : masks) {
                if (item.kind == ItemKind::face) {
                    append_cuda_label(
                        outline_spans_,
                        outline_batch_ends_,
                        item,
                        frame->width,
                        frame->height
                    );
                }
            }
        }
        std::array<char, 512> error{};
        const int result = cuda_overlay_apply_nv12(
            context_,
            frame->data[0],
            frame->data[1],
            frame->linesize[0],
            frame->linesize[1],
            frame->width,
            frame->height,
            fill_spans_.data(),
            fill_spans_.size(),
            fill_batch_ends_.data(),
            fill_batch_ends_.size(),
            outline_spans_.data(),
            outline_spans_.size(),
            outline_batch_ends_.data(),
            outline_batch_ends_.size(),
            error.data(),
            error.size()
        );
        if (result != 0) {
            throw std::runtime_error(
                std::string("CUDA overlay failed: ") + error.data()
            );
        }
    }

private:
    void* context_{nullptr};
    std::vector<CudaOverlaySpan> fill_spans_;
    std::vector<std::size_t> fill_batch_ends_;
    std::vector<CudaOverlaySpan> outline_spans_;
    std::vector<std::size_t> outline_batch_ends_;
};

Options parse_options(int argc, char** argv) {
    Options options;
    auto require_value = [&](int& index, std::string_view name) -> std::string {
        if (index + 1 >= argc) {
            throw std::runtime_error(
                "missing value for " + std::string(name)
            );
        }
        return argv[++index];
    };
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--video") {
            options.video = require_value(index, argument);
        } else if (argument == "--sqlite") {
            options.sqlite = require_value(index, argument);
        } else if (argument == "--face-sqlite") {
            options.face_sqlite = require_value(index, argument);
        } else if (argument == "--output") {
            options.output = require_value(index, argument);
        } else if (argument == "--manifest") {
            options.manifest = require_value(index, argument);
        } else if (argument == "--mode") {
            options.mode = require_value(index, argument);
        } else if (argument == "--include-faces") {
            options.include_faces = true;
        } else if (argument == "--no-labels") {
            options.show_labels = false;
        } else if (argument == "--copy-audio") {
            options.copy_audio = true;
        } else if (argument == "--encoder") {
            options.encoder = require_value(index, argument);
        } else if (argument == "--codec") {
            const std::string codec = require_value(index, argument);
            if (codec == "h264_nvenc" || codec == "nvenc") {
                options.encoder = "h264_nvenc";
            } else if (
                codec == "h264" ||
                codec == "avc1" ||
                codec == "x264" ||
                codec == "libx264"
            ) {
                options.encoder = "libx264";
            } else {
                throw std::runtime_error(
                    "low-level renderer supports only H.264 codecs"
                );
            }
        } else if (
            argument == "--preset" ||
            argument == "--nvenc-preset" ||
            argument == "--h264-preset"
        ) {
            options.preset = require_value(index, argument);
        } else if (argument == "--nvenc-gpu") {
            options.gpu_index =
                std::stoi(require_value(index, argument));
        } else if (
            argument == "--bitrate-mbps" ||
            argument == "--target-bitrate-mbps"
        ) {
            options.bitrate_mbps =
                std::stod(require_value(index, argument));
        } else if (
            argument == "--crf" ||
            argument == "--h264-crf"
        ) {
            options.crf = std::stoi(require_value(index, argument));
        } else if (argument == "--mask-alpha") {
            options.mask_alpha =
                std::stod(require_value(index, argument));
        } else if (argument == "--outline-thickness") {
            options.outline_thickness =
                std::stoi(require_value(index, argument));
        } else if (argument == "--box-thickness") {
            options.box_thickness =
                std::stoi(require_value(index, argument));
        } else if (argument == "--decoder-threads") {
            options.decoder_threads =
                std::stoi(require_value(index, argument));
        } else if (argument == "--hw-decode") {
            options.hw_decode = true;
        } else if (argument == "--gpu-pipeline") {
            options.gpu_pipeline = true;
        } else if (argument == "--faststart") {
            options.faststart = true;
        } else if (argument == "--start-frame") {
            options.start_frame =
                std::stoi(require_value(index, argument));
        } else if (argument == "--end-frame") {
            options.end_frame =
                std::stoi(require_value(index, argument));
        } else if (argument == "--overwrite") {
            options.overwrite = true;
        } else if (argument == "--help" || argument == "-h") {
            std::cout
                << "overlay_native --video FILE --sqlite FILE --output FILE "
                << "[--mode raw|tracked|final|faces] "
                << "[--include-faces --face-sqlite FILE] "
                << "[--no-labels] [--copy-audio] [--manifest FILE] "
                << "[--encoder libx264|h264_nvenc] [--preset VALUE] "
                << "[--codec h264|h264_nvenc] [--nvenc-gpu N] "
                << "[--bitrate-mbps N | --crf N] [--mask-alpha N] "
                << "[--outline-thickness N] [--box-thickness N] "
                << "[--decoder-threads N] "
                << "[--hw-decode] "
                << "[--gpu-pipeline] "
                << "[--faststart] "
                << "[--start-frame N] "
                << "[--end-frame N] [--overwrite]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + argument);
        }
    }
    if (options.video.empty() || options.sqlite.empty() || options.output.empty()) {
        throw std::runtime_error("--video, --sqlite and --output are required");
    }
    if (
        options.mode != "raw" &&
        options.mode != "tracked" &&
        options.mode != "final" &&
        options.mode != "faces"
    ) {
        throw std::runtime_error(
            "mode must be raw, tracked, final or faces"
        );
    }
    if (options.include_faces && options.mode != "final") {
        throw std::runtime_error(
            "--include-faces is only valid with --mode final"
        );
    }
    if (options.include_faces && options.face_sqlite.empty()) {
        throw std::runtime_error(
            "--include-faces requires --face-sqlite"
        );
    }
    if (!options.face_sqlite.empty() && !options.include_faces) {
        throw std::runtime_error(
            "--face-sqlite requires --mode final --include-faces"
        );
    }
    if (!options.manifest.empty() && options.manifest == options.output) {
        throw std::runtime_error("manifest must differ from output");
    }
    if (options.encoder != "libx264" && options.encoder != "h264_nvenc") {
        throw std::runtime_error("encoder must be libx264 or h264_nvenc");
    }
    if (options.bitrate_mbps <= 0.0) {
        throw std::runtime_error("bitrate must be positive");
    }
    if (options.crf && (*options.crf < 0 || *options.crf > 51)) {
        throw std::runtime_error("CRF must be between 0 and 51");
    }
    if (options.crf && options.encoder != "libx264") {
        throw std::runtime_error("--crf currently requires libx264");
    }
    if (options.hw_decode && options.encoder != "h264_nvenc") {
        throw std::runtime_error("--hw-decode currently requires h264_nvenc");
    }
    if (options.gpu_pipeline && options.encoder != "h264_nvenc") {
        throw std::runtime_error(
            "--gpu-pipeline currently requires h264_nvenc"
        );
    }
    if (options.gpu_pipeline) {
        options.hw_decode = true;
    }
    if (options.mask_alpha < 0.0 || options.mask_alpha > 1.0) {
        throw std::runtime_error("mask alpha must be between 0 and 1");
    }
    if (options.outline_thickness < 0) {
        throw std::runtime_error("outline thickness must be non-negative");
    }
    if (options.box_thickness < 0) {
        throw std::runtime_error("box thickness must be non-negative");
    }
    if (options.decoder_threads < 0) {
        throw std::runtime_error("decoder threads must be non-negative");
    }
    if (options.gpu_index < 0) {
        throw std::runtime_error("GPU index must be non-negative");
    }
    if (options.start_frame < 0) {
        throw std::runtime_error("start frame must be non-negative");
    }
    if (options.end_frame && *options.end_frame < options.start_frame) {
        throw std::runtime_error("end frame must be >= start frame");
    }
    return options;
}

struct FormatInputDeleter {
    void operator()(AVFormatContext* context) const {
        avformat_close_input(&context);
    }
};

struct CodecContextDeleter {
    void operator()(AVCodecContext* context) const {
        avcodec_free_context(&context);
    }
};

struct FrameDeleter {
    void operator()(AVFrame* frame) const { av_frame_free(&frame); }
};

struct PacketDeleter {
    void operator()(AVPacket* packet) const { av_packet_free(&packet); }
};

struct BufferRefDeleter {
    void operator()(AVBufferRef* reference) const {
        av_buffer_unref(&reference);
    }
};

using InputPtr = std::unique_ptr<AVFormatContext, FormatInputDeleter>;
using CodecPtr = std::unique_ptr<AVCodecContext, CodecContextDeleter>;
using FramePtr = std::unique_ptr<AVFrame, FrameDeleter>;
using PacketPtr = std::unique_ptr<AVPacket, PacketDeleter>;
using BufferRefPtr = std::unique_ptr<AVBufferRef, BufferRefDeleter>;

AVPixelFormat choose_cuda_format(
    AVCodecContext*,
    const AVPixelFormat* formats
) {
    for (const AVPixelFormat* format = formats;
         *format != AV_PIX_FMT_NONE;
         ++format) {
        if (*format == AV_PIX_FMT_CUDA) {
            return *format;
        }
    }
    return AV_PIX_FMT_NONE;
}

class OutputContext {
public:
    ~OutputContext() {
        if (context_) {
            if (context_->pb) {
                avio_closep(&context_->pb);
            }
            avformat_free_context(context_);
        }
    }

    AVFormatContext** address() { return &context_; }
    AVFormatContext* get() const { return context_; }

private:
    AVFormatContext* context_{nullptr};
};

struct RunSummary {
    int frames{};
    int mask_rows{};
    int face_rows{};
    int audio_packets{};
    int width{};
    int height{};
    double fps{};
    double elapsed_seconds{};
    double mask_seconds{};
    double encoder_seconds{};
    std::uintmax_t size_bytes{};
};

RunSummary render(const Options& options, const FrameMasks& masks) {
    auto gpu_trace = [&](const char* message) {
        if (options.gpu_pipeline && std::getenv("OVERLAY_CUDA_TRACE")) {
            std::cerr << "[cuda-trace] " << message << '\n';
        }
    };
    AVFormatContext* raw_input = nullptr;
    check_av(
        avformat_open_input(
            &raw_input,
            options.video.string().c_str(),
            nullptr,
            nullptr
        ),
        "open input"
    );
    InputPtr input(raw_input);
    check_av(avformat_find_stream_info(input.get(), nullptr), "stream info");
    const int video_stream_index = av_find_best_stream(
        input.get(),
        AVMEDIA_TYPE_VIDEO,
        -1,
        -1,
        nullptr,
        0
    );
    check_av(video_stream_index, "find video stream");
    AVStream* input_stream = input->streams[video_stream_index];

    const AVCodec* decoder =
        avcodec_find_decoder(input_stream->codecpar->codec_id);
    if (!decoder) {
        throw std::runtime_error("video decoder not found");
    }
    CodecPtr decoder_context(avcodec_alloc_context3(decoder));
    if (!decoder_context) {
        throw std::runtime_error("decoder allocation failed");
    }
    check_av(
        avcodec_parameters_to_context(
            decoder_context.get(),
            input_stream->codecpar
        ),
        "copy decoder parameters"
    );
    decoder_context->thread_count = options.decoder_threads;
    BufferRefPtr hardware_device;
    if (options.hw_decode) {
        AVBufferRef* raw_device = nullptr;
        AVDictionary* device_options = nullptr;
        av_dict_set(&device_options, "primary_ctx", "1", 0);
        const std::string device_name = std::to_string(options.gpu_index);
        const int device_result = av_hwdevice_ctx_create(
            &raw_device,
            AV_HWDEVICE_TYPE_CUDA,
            device_name.c_str(),
            device_options,
            0
        );
        av_dict_free(&device_options);
        check_av(device_result, "create CUDA decode device");
        hardware_device.reset(raw_device);
        decoder_context->hw_device_ctx = av_buffer_ref(
            hardware_device.get()
        );
        if (!decoder_context->hw_device_ctx) {
            throw std::runtime_error("CUDA device reference failed");
        }
        decoder_context->get_format = choose_cuda_format;
    }
    check_av(
        avcodec_open2(decoder_context.get(), decoder, nullptr),
        "open decoder"
    );
    gpu_trace("decoder opened");

    const AVRational frame_rate = av_guess_frame_rate(
        input.get(),
        input_stream,
        nullptr
    );
    if (frame_rate.num <= 0 || frame_rate.den <= 0) {
        throw std::runtime_error("input frame rate is unavailable");
    }
    const double source_fps = av_q2d(frame_rate);
    validate_sqlite_video_metadata(
        options.sqlite,
        decoder_context->width,
        decoder_context->height,
        source_fps
    );
    if (options.include_faces) {
        validate_sqlite_video_metadata(
            options.face_sqlite,
            decoder_context->width,
            decoder_context->height,
            source_fps
        );
    }
    if (
        !masks.empty() &&
        input_stream->nb_frames > 0 &&
        masks.rbegin()->first >= input_stream->nb_frames
    ) {
        throw std::runtime_error(
            "SQLite frame index exceeds input video frame count"
        );
    }
    int audio_stream_index = -1;
    AVStream* input_audio_stream = nullptr;
    if (options.copy_audio) {
        audio_stream_index = av_find_best_stream(
            input.get(),
            AVMEDIA_TYPE_AUDIO,
            -1,
            video_stream_index,
            nullptr,
            0
        );
        if (audio_stream_index >= 0) {
            input_audio_stream = input->streams[audio_stream_index];
        } else {
            audio_stream_index = -1;
        }
    }
    const std::int64_t audio_stream_start =
        input_audio_stream &&
            input_audio_stream->start_time != AV_NOPTS_VALUE
        ? input_audio_stream->start_time
        : 0;
    const std::int64_t audio_selection_start = input_audio_stream
        ? audio_stream_start +
            av_rescale_q(
                options.start_frame,
                av_inv_q(frame_rate),
                input_audio_stream->time_base
            )
        : 0;
    const std::optional<std::int64_t> audio_selection_end =
        input_audio_stream && options.end_frame
        ? std::optional<std::int64_t>(
              audio_stream_start +
              av_rescale_q(
                  static_cast<std::int64_t>(*options.end_frame) + 1,
                  av_inv_q(frame_rate),
                  input_audio_stream->time_base
              )
          )
        : std::nullopt;

    OutputContext output;
    check_av(
        avformat_alloc_output_context2(
            output.address(),
            nullptr,
            "mp4",
            options.output.string().c_str()
        ),
        "allocate output"
    );
    const AVCodec* encoder = avcodec_find_encoder_by_name(
        options.encoder.c_str()
    );
    if (!encoder) {
        throw std::runtime_error("encoder not found: " + options.encoder);
    }
    CodecPtr encoder_context(avcodec_alloc_context3(encoder));
    if (!encoder_context) {
        throw std::runtime_error("encoder allocation failed");
    }
    encoder_context->width = decoder_context->width;
    encoder_context->height = decoder_context->height;
    encoder_context->pix_fmt = (
        options.gpu_pipeline
            ? AV_PIX_FMT_CUDA
            : (options.hw_decode ? AV_PIX_FMT_NV12 : AV_PIX_FMT_YUV420P)
    );
    encoder_context->time_base = av_inv_q(frame_rate);
    encoder_context->framerate = frame_rate;
    encoder_context->sample_aspect_ratio =
        input_stream->codecpar->sample_aspect_ratio;
    encoder_context->color_range = decoder_context->color_range;
    encoder_context->color_primaries = decoder_context->color_primaries;
    encoder_context->color_trc = decoder_context->color_trc;
    encoder_context->colorspace = decoder_context->colorspace;
    if (!options.crf || *options.crf != 0) {
        encoder_context->profile = AV_PROFILE_H264_HIGH;
    }
    const auto bitrate = static_cast<std::int64_t>(
        std::llround(options.bitrate_mbps * 1'000'000.0)
    );
    if (!options.crf) {
        encoder_context->bit_rate = bitrate;
        encoder_context->rc_min_rate = bitrate;
        encoder_context->rc_max_rate = bitrate;
        encoder_context->rc_buffer_size = bitrate * 2;
    }
    if (output.get()->oformat->flags & AVFMT_GLOBALHEADER) {
        encoder_context->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }
    if (options.gpu_pipeline) {
        encoder_context->hw_device_ctx = av_buffer_ref(
            hardware_device.get()
        );
        if (!encoder_context->hw_device_ctx) {
            throw std::runtime_error(
                "CUDA encoder context reference failed"
            );
        }
    }

    AVStream* output_stream = nullptr;
    AVStream* output_audio_stream = nullptr;
    std::vector<PacketPtr> buffered_audio_packets;
    bool output_started = false;
    auto open_output = [&](AVFrame* first_frame) {
        if (output_started) {
            return;
        }
        if (options.gpu_pipeline) {
            if (!first_frame || !first_frame->hw_frames_ctx) {
                throw std::runtime_error(
                    "decoded CUDA frame has no hardware frames context"
                );
            }
            encoder_context->hw_frames_ctx = av_buffer_ref(
                first_frame->hw_frames_ctx
            );
            if (!encoder_context->hw_frames_ctx) {
                throw std::runtime_error(
                    "CUDA encoder frames context reference failed"
                );
            }
        }

        AVDictionary* encoder_options = nullptr;
        av_dict_set(&encoder_options, "preset", options.preset.c_str(), 0);
        if (!options.crf || *options.crf != 0) {
            av_dict_set(&encoder_options, "profile", "high", 0);
        }
        if (options.crf) {
            const std::string crf = std::to_string(*options.crf);
            av_dict_set(&encoder_options, "crf", crf.c_str(), 0);
        }
        if (options.encoder == "h264_nvenc") {
            const std::string gpu_index = std::to_string(
                options.gpu_index
            );
            av_dict_set(
                &encoder_options,
                "gpu",
                gpu_index.c_str(),
                0
            );
            av_dict_set(&encoder_options, "tune", "hq", 0);
            av_dict_set(&encoder_options, "rc", "cbr", 0);
            av_dict_set(&encoder_options, "multipass", "disabled", 0);
            av_dict_set(&encoder_options, "spatial-aq", "1", 0);
            av_dict_set(&encoder_options, "temporal-aq", "1", 0);
            av_dict_set(&encoder_options, "aq-strength", "8", 0);
            av_dict_set(&encoder_options, "cbr_padding", "1", 0);
        }
        const int encoder_open_result = avcodec_open2(
            encoder_context.get(),
            encoder,
            &encoder_options
        );
        if (encoder_open_result < 0) {
            av_dict_free(&encoder_options);
            check_av(encoder_open_result, "open encoder");
        }
        gpu_trace("encoder opened");
        if (av_dict_count(encoder_options) != 0) {
            const AVDictionaryEntry* entry = nullptr;
            while ((entry = av_dict_iterate(encoder_options, entry))) {
                std::cerr << "warning: unused encoder option: "
                          << entry->key << '=' << entry->value << '\n';
            }
        }
        av_dict_free(&encoder_options);

        output_stream = avformat_new_stream(output.get(), nullptr);
        if (!output_stream) {
            throw std::runtime_error("output stream allocation failed");
        }
        output_stream->time_base = encoder_context->time_base;
        output_stream->avg_frame_rate = frame_rate;
        check_av(
            avcodec_parameters_from_context(
                output_stream->codecpar,
                encoder_context.get()
            ),
            "copy encoder parameters"
        );
        if (input_audio_stream) {
            output_audio_stream = avformat_new_stream(
                output.get(),
                nullptr
            );
            if (!output_audio_stream) {
                throw std::runtime_error(
                    "output audio stream allocation failed"
                );
            }
            check_av(
                avcodec_parameters_copy(
                    output_audio_stream->codecpar,
                    input_audio_stream->codecpar
                ),
                "copy audio stream parameters"
            );
            output_audio_stream->codecpar->codec_tag = 0;
            output_audio_stream->time_base =
                input_audio_stream->time_base;
        }
        check_av(
            avio_open(
                &output.get()->pb,
                options.output.string().c_str(),
                AVIO_FLAG_WRITE
            ),
            "open output"
        );
        AVDictionary* mux_options = nullptr;
        if (options.faststart) {
            av_dict_set(&mux_options, "movflags", "+faststart", 0);
        }
        const int header_result =
            avformat_write_header(output.get(), &mux_options);
        av_dict_free(&mux_options);
        check_av(header_result, "write output header");
        output_started = true;
        gpu_trace("output header written");
    };

    if (!options.gpu_pipeline) {
        open_output(nullptr);
    }

    FramePtr frame(av_frame_alloc());
    FramePtr software_frame(av_frame_alloc());
    PacketPtr input_packet(av_packet_alloc());
    PacketPtr output_packet(av_packet_alloc());
    if (!frame || !software_frame || !input_packet || !output_packet) {
        throw std::runtime_error("frame/packet allocation failed");
    }
    std::unique_ptr<CudaOverlayContext> cuda_overlay;
    if (options.gpu_pipeline) {
        cuda_overlay = std::make_unique<CudaOverlayContext>(
            options.gpu_index
        );
    }

    int fallback_source_frame = 0;
    int encoded_frames = 0;
    int mask_rows = 0;
    int face_rows = 0;
    int audio_packets = 0;
    bool reached_end = false;
    double mask_seconds = 0.0;
    double encoder_seconds = 0.0;

    auto write_audio_packet = [&](AVPacket* packet) {
        if (!output_audio_stream || !input_audio_stream) {
            return;
        }
        av_packet_rescale_ts(
            packet,
            input_audio_stream->time_base,
            output_audio_stream->time_base
        );
        packet->stream_index = output_audio_stream->index;
        packet->pos = -1;
        check_av(
            av_interleaved_write_frame(output.get(), packet),
            "write audio packet"
        );
        ++audio_packets;
    };

    auto flush_buffered_audio = [&]() {
        if (!output_started) {
            return;
        }
        for (auto& packet : buffered_audio_packets) {
            write_audio_packet(packet.get());
        }
        buffered_audio_packets.clear();
    };

    auto accept_audio_packet = [&](const AVPacket* packet) {
        if (!input_audio_stream) {
            return false;
        }
        const std::int64_t timestamp =
            packet->pts != AV_NOPTS_VALUE ? packet->pts : packet->dts;
        if (timestamp == AV_NOPTS_VALUE) {
            return false;
        }
        const std::int64_t packet_end =
            timestamp + std::max<std::int64_t>(packet->duration, 1);
        if (packet_end <= audio_selection_start) {
            return false;
        }
        if (audio_selection_end && timestamp >= *audio_selection_end) {
            return false;
        }
        return true;
    };

    auto queue_or_write_audio = [&](const AVPacket* source) {
        if (!accept_audio_packet(source)) {
            return;
        }
        PacketPtr packet(av_packet_clone(source));
        if (!packet) {
            throw std::runtime_error("audio packet clone failed");
        }
        if (packet->pts != AV_NOPTS_VALUE) {
            packet->pts -= audio_selection_start;
        }
        if (packet->dts != AV_NOPTS_VALUE) {
            packet->dts -= audio_selection_start;
        }
        if (output_started) {
            write_audio_packet(packet.get());
        } else {
            buffered_audio_packets.push_back(std::move(packet));
        }
    };

    if (options.start_frame > 0) {
        const std::int64_t start_timestamp =
            av_rescale_q(
                options.start_frame,
                av_inv_q(frame_rate),
                input_stream->time_base
            ) +
            (
                input_stream->start_time == AV_NOPTS_VALUE
                    ? 0
                    : input_stream->start_time
            );
        check_av(
            av_seek_frame(
                input.get(),
                video_stream_index,
                start_timestamp,
                AVSEEK_FLAG_BACKWARD
            ),
            "seek input"
        );
        avcodec_flush_buffers(decoder_context.get());
        fallback_source_frame = options.start_frame;
    }

    auto drain_encoder = [&]() {
        while (true) {
            const int result = avcodec_receive_packet(
                encoder_context.get(),
                output_packet.get()
            );
            if (result == AVERROR(EAGAIN) || result == AVERROR_EOF) {
                return;
            }
            check_av(result, "receive encoded packet");
            av_packet_rescale_ts(
                output_packet.get(),
                encoder_context->time_base,
                output_stream->time_base
            );
            output_packet->stream_index = output_stream->index;
            check_av(
                av_interleaved_write_frame(output.get(), output_packet.get()),
                "write encoded packet"
            );
            av_packet_unref(output_packet.get());
        }
    };

    auto process_decoded_frames = [&]() {
        while (true) {
            const int result =
                avcodec_receive_frame(decoder_context.get(), frame.get());
            if (result == AVERROR(EAGAIN) || result == AVERROR_EOF) {
                return;
            }
            check_av(result, "receive decoded frame");
            gpu_trace("decoded frame received");
            AVFrame* processing_frame = frame.get();
            if (
                frame->format == AV_PIX_FMT_CUDA &&
                !options.gpu_pipeline
            ) {
                av_frame_unref(software_frame.get());
                check_av(
                    av_hwframe_transfer_data(
                        software_frame.get(),
                        frame.get(),
                        0
                    ),
                    "download CUDA frame"
                );
                check_av(
                    av_frame_copy_props(software_frame.get(), frame.get()),
                    "copy CUDA frame properties"
                );
                processing_frame = software_frame.get();
            }
            if (
                processing_frame->format != AV_PIX_FMT_YUV420P &&
                processing_frame->format != AV_PIX_FMT_NV12 &&
                processing_frame->format != AV_PIX_FMT_CUDA
            ) {
                throw std::runtime_error(
                    "native renderer requires yuv420p, NV12 or CUDA input"
                );
            }
            int source_frame = fallback_source_frame;
            if (frame->best_effort_timestamp != AV_NOPTS_VALUE) {
                const std::int64_t relative_timestamp =
                    frame->best_effort_timestamp -
                    (
                        input_stream->start_time == AV_NOPTS_VALUE
                            ? 0
                            : input_stream->start_time
                    );
                source_frame = static_cast<int>(
                    av_rescale_q_rnd(
                        relative_timestamp,
                        input_stream->time_base,
                        av_inv_q(frame_rate),
                        static_cast<AVRounding>(
                            AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX
                        )
                    )
                );
            }
            fallback_source_frame = source_frame + 1;
            if (options.end_frame && source_frame > *options.end_frame) {
                reached_end = true;
                av_frame_unref(frame.get());
                return;
            }
            if (source_frame >= options.start_frame) {
                check_av(
                    av_frame_make_writable(processing_frame),
                    "writable frame"
                );
                gpu_trace("frame writable");
                const auto mask_iterator = masks.find(source_frame);
                if (mask_iterator != masks.end()) {
                    const auto mask_started =
                        std::chrono::steady_clock::now();
                    if (options.gpu_pipeline) {
                        cuda_overlay->apply(
                            processing_frame,
                            mask_iterator->second,
                            options.mask_alpha,
                            options.outline_thickness,
                            options.box_thickness,
                            options.show_labels
                        );
                    } else {
                        draw_masks(
                            processing_frame,
                            mask_iterator->second,
                            options.mask_alpha,
                            options.outline_thickness,
                            options.box_thickness,
                            options.show_labels
                        );
                    }
                    const auto mask_completed =
                        std::chrono::steady_clock::now();
                    mask_seconds += std::chrono::duration<double>(
                        mask_completed - mask_started
                    ).count();
                    for (const auto& item : mask_iterator->second) {
                        if (item.kind == ItemKind::mask) {
                            ++mask_rows;
                        } else {
                            ++face_rows;
                        }
                    }
                }
                processing_frame->pts = encoded_frames;
                processing_frame->duration = 1;
                if (!output_started) {
                    open_output(processing_frame);
                    flush_buffered_audio();
                }
                const auto encoder_started =
                    std::chrono::steady_clock::now();
                check_av(
                    avcodec_send_frame(
                        encoder_context.get(),
                        processing_frame
                    ),
                    "send frame to encoder"
                );
                gpu_trace("frame sent to encoder");
                drain_encoder();
                const auto encoder_completed =
                    std::chrono::steady_clock::now();
                encoder_seconds += std::chrono::duration<double>(
                    encoder_completed - encoder_started
                ).count();
                ++encoded_frames;
            }
            av_frame_unref(software_frame.get());
            av_frame_unref(frame.get());
        }
    };

    while (!reached_end && av_read_frame(input.get(), input_packet.get()) >= 0) {
        if (input_packet->stream_index == video_stream_index) {
            check_av(
                avcodec_send_packet(
                    decoder_context.get(),
                    input_packet.get()
                ),
                "send packet to decoder"
            );
            process_decoded_frames();
        } else if (input_packet->stream_index == audio_stream_index) {
            queue_or_write_audio(input_packet.get());
        }
        av_packet_unref(input_packet.get());
    }
    if (!reached_end) {
        check_av(
            avcodec_send_packet(decoder_context.get(), nullptr),
            "flush decoder"
        );
        process_decoded_frames();
    }
    if (output_started) {
        check_av(
            avcodec_send_frame(encoder_context.get(), nullptr),
            "flush encoder"
        );
        drain_encoder();
        check_av(av_write_trailer(output.get()), "write trailer");
    } else {
        throw std::runtime_error("no video frames were selected");
    }

    const double fps = av_q2d(frame_rate);
    return RunSummary{
        encoded_frames,
        mask_rows,
        face_rows,
        audio_packets,
        decoder_context->width,
        decoder_context->height,
        fps,
        0.0,
        mask_seconds,
        encoder_seconds,
        0,
    };
}

std::string json_escape(const std::string& value) {
    std::ostringstream output;
    for (const char character : value) {
        switch (character) {
        case '\\':
            output << "\\\\";
            break;
        case '"':
            output << "\\\"";
            break;
        case '\n':
            output << "\\n";
            break;
        case '\r':
            output << "\\r";
            break;
        case '\t':
            output << "\\t";
            break;
        default:
            output << character;
        }
    }
    return output.str();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        av_log_set_level(AV_LOG_ERROR);
        const Options options = parse_options(argc, argv);
        if (!fs::is_regular_file(options.video)) {
            throw std::runtime_error("input video not found");
        }
        if (!fs::is_regular_file(options.sqlite)) {
            throw std::runtime_error("input SQLite not found");
        }
        if (
            options.include_faces &&
            !fs::is_regular_file(options.face_sqlite)
        ) {
            throw std::runtime_error("face SQLite not found");
        }
        if (options.output == options.video) {
            throw std::runtime_error("output must differ from input");
        }
        if (fs::exists(options.output) && !options.overwrite) {
            throw std::runtime_error(
                "output exists; pass --overwrite to replace it"
            );
        }
        if (
            !options.manifest.empty() &&
            fs::exists(options.manifest) &&
            !options.overwrite
        ) {
            throw std::runtime_error(
                "manifest exists; pass --overwrite to replace it"
            );
        }
        if (!options.output.parent_path().empty()) {
            fs::create_directories(options.output.parent_path());
        }
        if (
            !options.manifest.empty() &&
            !options.manifest.parent_path().empty()
        ) {
            fs::create_directories(options.manifest.parent_path());
        }

        const auto started = std::chrono::steady_clock::now();
        FrameMasks masks;
        if (options.mode == "raw") {
            masks = load_raw_masks(
                options.sqlite,
                options.start_frame,
                options.end_frame
            );
        } else if (
            options.mode == "tracked" ||
            options.mode == "final"
        ) {
            masks = load_postprocess_masks(
                options.sqlite,
                options.start_frame,
                options.end_frame
            );
        } else {
            masks = load_faces(
                options.sqlite,
                options.start_frame,
                options.end_frame
            );
        }
        if (options.include_faces) {
            merge_frame_items(
                masks,
                load_faces(
                    options.face_sqlite,
                    options.start_frame,
                    options.end_frame
                )
            );
        }
        const std::string unique_suffix = std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()
        );
        const fs::path temporary_output = options.output.parent_path() /
            (
                "." + options.output.stem().string() + "." +
                unique_suffix + ".tmp" + options.output.extension().string()
            );
        Options render_options = options;
        render_options.output = temporary_output;
        RunSummary summary;
        try {
            summary = render(render_options, masks);
            fs::rename(temporary_output, options.output);
        } catch (...) {
            std::error_code ignored;
            fs::remove(temporary_output, ignored);
            throw;
        }
        const auto completed = std::chrono::steady_clock::now();
        summary.elapsed_seconds =
            std::chrono::duration<double>(completed - started).count();
        summary.size_bytes = fs::file_size(options.output);

        std::ostringstream json;
        json << std::fixed << std::setprecision(6)
             << "{\n"
                  << "  \"implementation\": \""
                  << (
                         options.gpu_pipeline
                             ? "cpp-libav-nvdec-cuda-nvenc"
                             : "cpp-libav-yuv420p"
                     )
                  << "\",\n"
                  << "  \"video\": \"" << json_escape(options.video.string())
                  << "\",\n"
                  << "  \"mode\": \"" << options.mode << "\",\n"
                  << "  \"sqlite\": \"" << json_escape(options.sqlite.string())
                  << "\",\n"
                  << "  \"include_faces\": "
                  << (options.include_faces ? "true" : "false") << ",\n"
                  << "  \"show_labels\": "
                  << (options.show_labels ? "true" : "false") << ",\n"
                  << "  \"output\": \"" << json_escape(options.output.string())
                  << "\",\n"
                  << "  \"encoder\": \"" << options.encoder << "\",\n"
                  << "  \"preset\": \"" << options.preset << "\",\n"
                  << "  \"gpu_index\": " << options.gpu_index << ",\n"
                  << "  \"target_bitrate_mbps\": " << options.bitrate_mbps
                  << ",\n"
                  << "  \"crf\": ";
        if (options.crf) {
            json << *options.crf;
        } else {
            json << "null";
        }
        json << ",\n"
             << "  \"decoder_threads\": " << options.decoder_threads
                  << ",\n"
                  << "  \"hw_decode\": "
                  << (options.hw_decode ? "true" : "false") << ",\n"
                  << "  \"gpu_pipeline\": "
                  << (options.gpu_pipeline ? "true" : "false") << ",\n"
                  << "  \"faststart\": "
                  << (options.faststart ? "true" : "false") << ",\n"
                  << "  \"width\": " << summary.width << ",\n"
                  << "  \"height\": " << summary.height << ",\n"
                  << "  \"source_fps\": " << summary.fps << ",\n"
                  << "  \"frames_written\": " << summary.frames << ",\n"
                  << "  \"mask_rows_drawn\": " << summary.mask_rows << ",\n"
                  << "  \"face_rows_drawn\": " << summary.face_rows << ",\n"
                  << "  \"audio_packets_copied\": "
                  << summary.audio_packets << ",\n"
                  << "  \"audio_copied\": "
                  << (summary.audio_packets > 0 ? "true" : "false")
                  << ",\n"
                  << "  \"elapsed_seconds\": " << summary.elapsed_seconds
                  << ",\n"
                  << "  \"mask_seconds\": " << summary.mask_seconds
                  << ",\n"
                  << "  \"encoder_seconds\": " << summary.encoder_seconds
                  << ",\n"
                  << "  \"other_seconds\": "
                  << std::max(
                         0.0,
                         summary.elapsed_seconds - summary.mask_seconds -
                             summary.encoder_seconds
                     )
                  << ",\n"
                  << "  \"aggregate_fps\": "
                  << summary.frames / summary.elapsed_seconds << ",\n"
                  << "  \"size_bytes\": " << summary.size_bytes << "\n"
                  << "}\n";
        const std::string encoded_json = json.str();
        std::cout << encoded_json;
        if (!options.manifest.empty()) {
            const fs::path temporary_manifest =
                options.manifest.parent_path() /
                (
                    "." + options.manifest.filename().string() + "." +
                    unique_suffix + ".tmp"
                );
            try {
                {
                    std::ofstream output_manifest(
                        temporary_manifest,
                        std::ios::binary | std::ios::trunc
                    );
                    if (!output_manifest) {
                        throw std::runtime_error(
                            "failed to open temporary manifest"
                        );
                    }
                    output_manifest << encoded_json;
                    if (!output_manifest) {
                        throw std::runtime_error(
                            "failed to write temporary manifest"
                        );
                    }
                }
                fs::rename(temporary_manifest, options.manifest);
            } catch (...) {
                std::error_code ignored;
                fs::remove(temporary_manifest, ignored);
                throw;
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
