#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <omp.h>

namespace py = pybind11;

std::tuple<py::array_t<float>, py::array_t<uint8_t>, py::array_t<int8_t>> inverse_projection(
    py::array_t<float> K,
    py::array_t<float> depth_image,
    py::array_t<uint8_t> rgb_image,
    py::array_t<int8_t> instance_mask = py::array_t<int8_t>()) {
    
    // 获取输入数组的形状
    auto depth_buf = depth_image.request();
    auto rgb_buf = rgb_image.request();
    auto K_buf = K.request();
    
    int height = depth_buf.shape[0];
    int width = depth_buf.shape[1];
    
    // 从 numpy 数组获取指针
    float *depth_ptr = static_cast<float *>(depth_buf.ptr);
    uint8_t *rgb_ptr = static_cast<uint8_t *>(rgb_buf.ptr);
    float *K_ptr = static_cast<float *>(K_buf.ptr);
    int8_t *instance_ptr = nullptr;

    if (!instance_mask.is_none()) {
        auto instance_buf = instance_mask.request();
        instance_ptr = static_cast<int8_t *>(instance_buf.ptr);
    }

    // 输出结果的数组
    std::vector<float> points(height * width * 3);
    std::vector<uint8_t> colors(height * width * 3);
    std::vector<int8_t> ids;

    if (instance_ptr) {
        ids.resize(height * width);
    }

    // 相机内参
    float fx = K_ptr[0];
    float fy = K_ptr[4];
    float cx = K_ptr[2];
    float cy = K_ptr[5];

    // 逆向投影过程
    #pragma omp parallel for
    for (int v = 0; v < height; ++v) {
        for (int u = 0; u < width; ++u) {
            int idx = v * width + u;
            float Z = depth_ptr[idx]/5000.0;

            if (Z > 0) {
                float x_n = (u - cx) / fx;
                float y_n = (v - cy) / fy;

                // 计算 3D 坐标
                points[idx * 3 + 0] = x_n * Z;
                points[idx * 3 + 1] = y_n * Z;
                points[idx * 3 + 2] = Z;

                // 获取颜色
                colors[idx * 3 + 0] = rgb_ptr[idx * 3 + 0];
                colors[idx * 3 + 1] = rgb_ptr[idx * 3 + 1];
                colors[idx * 3 + 2] = rgb_ptr[idx * 3 + 2];

                // 实例 ID
                if (instance_ptr) {
                    ids[idx] = instance_ptr[idx];
                }
            }
        }
    }

    // 返回 numpy 数组
    py::array_t<float> points_array({height * width, 3}, points.data());
    py::array_t<uint8_t> colors_array({height * width, 3}, colors.data());
    py::array_t<int8_t> ids_array;
    
    if (instance_ptr) {
        ids_array = py::array_t<int8_t>({height * width}, ids.data());
    }
    
    return std::make_tuple(points_array, colors_array, ids_array);
// 
    // 过滤掉全为0的点
    // std::vector<float> filtered_points;
    // std::vector<uint8_t> filtered_colors;
    // std::vector<int8_t> filtered_ids;

    // // 并行化过滤操作
    // #pragma omp parallel for
    // for (size_t i = 0; i < points.size(); i += 3) {
    //     if (!(points[i] == 0.0f && points[i + 1] == 0.0f && points[i + 2] == 0.0f)) {
    //         #pragma omp critical
    //         {
    //             filtered_points.push_back(points[i]);
    //             filtered_points.push_back(points[i + 1]);
    //             filtered_points.push_back(points[i + 2]);
    //             filtered_colors.push_back(colors[i]);
    //             filtered_colors.push_back(colors[i + 1]);
    //             filtered_colors.push_back(colors[i + 2]);
    //             if (!ids.empty()) {
    //                 filtered_ids.push_back(ids[i / 3]);
    //             }
    //         }
    //     }
    // }

    // auto points_array = py::array_t<float>(
    //     {static_cast<py::ssize_t>(filtered_points.size() / 3), 3},
    //     filtered_points.data()
    // );

    // auto colors_array = py::array_t<uint8_t>(
    //     {static_cast<py::ssize_t>(filtered_colors.size() / 3), 3},
    //     filtered_colors.data()
    // );

    // py::array_t<int8_t> ids_array;
    // if (!filtered_ids.empty()) {
    //     ids_array = py::array_t<int8_t>(
    //         {static_cast<py::ssize_t>(filtered_ids.size())},
    //         filtered_ids.data()
    //     );
    // }

    // return std::make_tuple(points_array, colors_array, ids_array);
}

PYBIND11_MODULE(inverse_projection_cpp, m) {
    m.def("inverse_projection", &inverse_projection, "Inverse projection from depth to point cloud",
          py::arg("K"), py::arg("depth_image"), py::arg("rgb_image"), py::arg("instance_mask") = py::none());
}
