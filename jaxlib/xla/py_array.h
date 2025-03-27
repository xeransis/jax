/* Copyright 2022 The JAX Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef JAXLIB_XLA_PY_ARRAY_H_
#define JAXLIB_XLA_PY_ARRAY_H_

#include <Python.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

// placeholder for index annotation headers
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/Support/Casting.h"
#include "nanobind/nanobind.h"
#include "jaxlib/xla/nb_class_ptr.h"
#include "jaxlib/xla/py_client.h"
#include "jaxlib/xla/traceback.h"
#include "xla/pjrt/exceptions.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/nb_numpy.h"
#include "xla/python/pjrt_ifrt/pjrt_array.h"
#include "xla/shape.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/util.h"

namespace xla {

// Private to PyArray, but you cannot forward declare member classes.
// Not thread safe; assumes the GIL is held.
class PyHostValue {
 public:
  PyHostValue();
  ~PyHostValue();

  PyHostValue(const PyHostValue&) = delete;
  PyHostValue(PyHostValue&&) = delete;
  PyHostValue& operator=(const PyHostValue&) = delete;
  PyHostValue& operator=(PyHostValue&&) = delete;

  absl::Status CopyToHostAsync(std::optional<Shape>& dynamic_shape_holder,
                               ifrt::Array* ifrt_array);

  absl::StatusOr<std::pair<nanobind::object, bool>> AsNumPyArray(
      std::optional<Shape>& dynamic_shape_holder, ifrt::Array* ifrt_array);

 private:
  absl::Status CopyStringArrayToHostAsync(
      std::optional<Shape>& dynamic_shape_holder, ifrt::Array* ifrt_array);

  absl::Status ConvertStringArrayContentsToNumpyArray(ifrt::Array* ifrt_array);

  ifrt::Future<> ready_;
  nb_numpy_ndarray value_;

  // Optional field, only used for arrays of type kString. This vector of cords
  // serves as input buffer for the CopyToHostBuffer call. It holds these
  // contents until it is lazily converted it to a numpy array when the user
  // calls `AsNumPyArray`.
  std::shared_ptr<std::vector<absl::Cord>> string_array_contents_;
};

// Private to PyArray, but you cannot forward declare member classes.
struct PyArray_Storage {
  PyArray_Storage(nanobind::object aval, bool weak_type, nb_dtype dtype,
                  std::vector<int64_t> shape, nanobind::object sharding,
                  bool committed, nb_class_ptr<PyClient> py_client,
                  std::optional<nb_traceback> traceback,
                  tsl::RCReference<ifrt::Array> ifrt_array,
                  xla::PjRtFuture<> result_status);

  ~PyArray_Storage();
  nanobind::handle AsHandle();

  nanobind::object aval;
  bool weak_type = false;
  nb_dtype dtype;
  std::vector<int64_t> shape;

  nanobind::object sharding;
  nanobind::object npy_value = nanobind::none();
  bool committed = false;

  nb_class_ptr<PyClient> py_client;
  std::optional<nb_traceback> traceback;
  tsl::RCReference<ifrt::Array> ifrt_array;
  nanobind::object fully_replicated_array = nanobind::none();

  // optional field, used only in python
  std::vector<PyArray> py_arrays;
  PyHostValue host_value;  // Protected by the GIL.
  std::optional<Shape> dynamic_shape = std::nullopt;
  // Only set if this Array was generated by a computation that has effects.
  // This is the result status of the XLA computation that generated this
  // array.
  xla::PjRtFuture<> result_status;

  // Doubly-linked list of all PyArrays known to the client. Protected by the
  // GIL. Since multiple PyArrays may share the same PjRtBuffer, there may be
  // duplicate PjRtBuffers in this list.
  PyArray_Storage* next;
  PyArray_Storage* prev;

  uint8_t thread_id_bucket;
};

// The C++ implementation of jax.Array. A few key methods and data members are
// implemented in C++ for performance, while most of the functionalities are
// still implemented in python.
class PyArray : public nanobind::object {
 public:
  NB_OBJECT(PyArray, nanobind::object, "Array", PyArray::IsPyArray);
  PyArray() = default;

  // "__init__" methods. Only used in python
  static void PyInit(PyArray self, nanobind::object aval,
                     nanobind::object sharding,
                     absl::Span<const PyArray> py_arrays, bool committed,
                     bool skip_checks);

  // Only used in C++. `skip_checks` should only be set for Arrays created by
  // jax that cannot possibly have consistency issues (e.g. `sharding` devices
  // different than `ifrt_array` devices). Arrays created by users should be
  // checked.
  PyArray(nanobind::object aval, bool weak_type, nb_dtype dtype,
          std::vector<int64_t> shape, nanobind::object sharding,
          nb_class_ptr<PyClient> py_client,
          std::optional<nb_traceback> traceback,
          tsl::RCReference<ifrt::Array> ifrt_array, bool committed,
          bool skip_checks,
          xla::PjRtFuture<> result_status = xla::PjRtFuture<>());

  static PyArray MakeFromSingleDeviceArray(
      nb_class_ptr<PyClient> py_client, std::optional<nb_traceback> traceback,
      tsl::RCReference<ifrt::Array> ifrt_array, bool weak_type, bool committed,
      xla::PjRtFuture<> result_status = xla::PjRtFuture<>());

  static PyArray MakeFromIfrtArrayAndSharding(
      nb_class_ptr<PyClient> py_client, std::optional<nb_traceback> traceback,
      tsl::RCReference<ifrt::Array> ifrt_array, nanobind::object sharding,
      bool weak_type, bool committed, bool skip_checks);

  static absl::Status RegisterTypes(nanobind::module_& m);

  static PyArray borrow(PyObject* ptr) {
    return nanobind::borrow<xla::PyArray>(ptr);
  }

  using Storage = PyArray_Storage;

  const nanobind::object& aval() const { return GetStorage().aval; }
  void set_aval(nanobind::object aval) { GetStorage().aval = std::move(aval); }

  bool weak_type() const { return GetStorage().weak_type; }

  const nb_dtype& dtype() const { return GetStorage().dtype; }
  absl::Span<const int64_t> shape() const { return GetStorage().shape; }

  const nanobind::object& sharding() const { return GetStorage().sharding; }

  absl::StatusOr<std::shared_ptr<const PjRtLayout>> layout() {
    return ifrt_array()->layout();
  }

  bool committed() const { return GetStorage().committed; }

  const nanobind::object& npy_value() const { return GetStorage().npy_value; }
  void set_npy_value(nanobind::object v) {
    GetStorage().npy_value = std::move(v);
  }

  const nb_class_ptr<PyClient>& py_client() const {
    return GetStorage().py_client;
  }

  const std::optional<nb_traceback>& traceback() const {
    return GetStorage().traceback;
  }

  // Returns xla::InvalidArgument if the buffer has been deleted.
  // See `PjRtFuture` for the semantics of `IsReady` and `IsKnownReady`.
  absl::StatusOr<bool> IsReady() {
    ifrt::Array* ifrt_array_ptr = ifrt_array();
    if (ifrt_array_ptr->IsDeleted()) {
      return InvalidArgument("Array has been deleted.");
    }
    return ifrt_array_ptr->GetReadyFuture().IsReady();
  }

  const xla::PjRtFuture<>& result_status() const {
    return GetStorage().result_status;
  }

  ifrt::Array* ifrt_array() const { return GetStorage().ifrt_array.get(); }

  // Short-term escape hatch to get PjRtBuffers from PyArray.
  // TODO(hyeontaek): Migrate all users of this method to be agnostic of PjRt.
  absl::Span<const std::shared_ptr<PjRtBuffer>> pjrt_buffers() const {
    ifrt::Array* ifrt_array_ptr = ifrt_array();
    if (ifrt_array_ptr == nullptr) {
      return {};
    }
    auto* arr =
        llvm::dyn_cast_or_null<ifrt::PjRtCompatibleArray>(ifrt_array_ptr);
    if (arr == nullptr) {
      throw XlaRuntimeError(
          "This operation is implemented for a PjRt-compatible backend only.");
    }
    return arr->pjrt_buffers();
  }

  int num_addressable_shards() const {
    ifrt::Array* ifrt_array_ptr = ifrt_array();
    if (ifrt_array_ptr == nullptr) {
      return 0;
    }
    auto* arr =
        llvm::dyn_cast_or_null<ifrt::PjRtCompatibleArray>(ifrt_array_ptr);
    if (arr == nullptr) {
      // TODO(hyeontaek): Add num_addressable_shards to ifrt.
      return num_shards();
    }
    return arr->pjrt_buffers().size();
  }

  std::vector<PyArray>& py_arrays() { return GetStorage().py_arrays; }
  const std::vector<PyArray>& py_arrays() const {
    return GetStorage().py_arrays;
  }
  const std::vector<PyArray>& py_arrays_cached();

  nanobind::object arrays();
  absl::Status set_arrays(nanobind::object obj);
  absl::StatusOr<PyArray> FullyReplicatedShard();

  int num_shards() const {
    ifrt::Array* ifrt_array_ptr = ifrt_array();
    if (ifrt_array_ptr == nullptr) {
      return 0;
    }
    return ifrt_array_ptr->sharding().devices()->size();
  }

  static nanobind::handle type() {
    DCHECK(type_);
    return nanobind::handle(type_);
  }

  static bool IsPyArray(nanobind::handle arg) {
    return arg.type().is(PyArray::type());
  }

  absl::Status BlockUntilReady() const;

  absl::Status BlockUntilResultStatusIsReady();

  absl::StatusOr<size_t> GetOnDeviceSizeInBytes();
  absl::StatusOr<std::pair<nanobind::object, bool>>
  SingleDeviceArrayToNumpyArrayDidCopy();
  absl::StatusOr<nanobind::object> SingleDeviceArrayToNumpyArray();
  absl::Status CopySingleDeviceArrayToHostAsync();
  nanobind::dict CudaArrayInterface();
  absl::StatusOr<std::uintptr_t> UnsafeBufferPointer();

  absl::Status Delete();

  bool IsDeleted() const;

  PyArray Clone() const;

  static absl::StatusOr<std::vector<PyArray>> BatchedCopyToDeviceWithSharding(
      absl::Span<const PyArray> py_arrays,
      absl::Span<const ifrt::DeviceListRef> dst_device_lists,
      absl::Span<const nanobind::object> dst_shardings,
      absl::Span<const ifrt::ArrayCopySemantics> array_copy_semantics);

  static absl::StatusOr<PyArray> BatchedDevicePut(
      nanobind::object aval, nanobind::object sharding,
      std::vector<nanobind::object> xs,
      absl::Span<const PyDevice* const> dst_devices, bool committed,
      bool force_copy, PjRtClient::HostBufferSemantics host_buffer_semantics,
      bool jax_enable_x64);

  static absl::StatusOr<PyArray> ReorderShards(
      PyArray x, nanobind::object dst_sharding,
      ifrt::ArrayCopySemantics array_copy_semantics);

  static absl::Status BatchedBlockUntilReady(
      std::vector<nanobind::object> objs);

 private:
  absl::StatusOr<PyArray> AssertUnsharded(absl::string_view api);

  nanobind::object CheckAndRearrange(absl::Span<const PyArray> py_arrays,
                                     nanobind::object sharding,
                                     nanobind::object aval);

  void SetIfrtArray(tsl::RCReference<ifrt::Array> ifrt_array);

  Storage& GetStorage();
  const Storage& GetStorage() const;

  inline static PyObject* type_ = nullptr;
};

class PyArrayResultHandler {
 public:
  PyArrayResultHandler(nanobind::object aval, nanobind::object sharding,
                       bool committed, bool skip_checks);

  PyArray Call(absl::Span<const PyArray> py_arrays) const;
  PyArray Call(PyArray py_array) const;

  PyArray Call(nb_class_ptr<PyClient> py_client,
               tsl::RCReference<ifrt::Array> ifrt_array,
               xla::PjRtFuture<> result_status = xla::PjRtFuture<>()) const;

 private:
  nanobind::object aval_;
  nanobind::object sharding_;
  bool weak_type_;
  bool committed_;
  bool skip_checks_;

  nb_dtype dtype_;
  std::vector<int64_t> shape_;
};

absl::StatusOr<nanobind::object> CudaArrayInterfaceToBuffer(
    const nanobind::dict& cai, nb_class_ptr<PyClient> cuda_client,
    std::optional<int> device_id);

}  // namespace xla

#endif  // JAXLIB_XLA_PY_ARRAY_H_
