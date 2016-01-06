#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
DataLayer<Dtype>::DataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param),
    reader_(param) {
}

template <typename Dtype>
DataLayer<Dtype>::~DataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void DataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.data_param().batch_size();
  // Read a data point, and use it to initialize the top blob.
  //<<<<<<< HEAD
  Datum& datum = *(reader_.full().peek());

  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
    //=======
//   Datum datum;
//   datum.ParseFromString(cursor_->value());

//   bool force_color = this->layer_param_.data_param().force_encoded_color();
//   if ((force_color && DecodeDatum(&datum, true)) ||
//       DecodeDatumNative(&datum)) {
//     LOG(INFO) << "Decoding Datum";
//   }
//   // image
//   const int crop_size = this->layer_param_.transform_param().crop_size();
//   int crop_h = this->layer_param_.transform_param().crop_h();
//   int crop_w = this->layer_param_.transform_param().crop_w();
//   if (crop_size > 0) {
//     crop_h = crop_w = crop_size;
//   }
//   if (crop_h > 0 || crop_w > 0) {
//     top[0]->Reshape(this->layer_param_.data_param().batch_size(),
//         datum.channels(), crop_h, crop_w);
//     this->prefetch_data_.Reshape(this->layer_param_.data_param().batch_size(),
//         datum.channels(), crop_h, crop_w);
//     this->transformed_data_.Reshape(1, datum.channels(), crop_h, crop_w);
//   } else {
//     top[0]->Reshape(
//         this->layer_param_.data_param().batch_size(), datum.channels(),
//         datum.height(), datum.width());
//     this->prefetch_data_.Reshape(this->layer_param_.data_param().batch_size(),
//         datum.channels(), datum.height(), datum.width());
//     this->transformed_data_.Reshape(1, datum.channels(),
//       datum.height(), datum.width());
// >>>>>>> Non-square
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  if (this->output_labels_) {
    vector<int> label_shape(1, batch_size);
    top[1]->Reshape(label_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(label_shape);
    }
  }
}

// This function is called on prefetch thread
template<typename Dtype>
void DataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());

  // Reshape according to the first datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  //<<<<<<< HEAD
  Datum& datum = *(reader_.full().peek());
  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* top_data = batch->data_.mutable_cpu_data();
// =======
//   const int crop_size = this->layer_param_.transform_param().crop_size();
//   const int crop_h = this->layer_param_.transform_param().crop_h();
//   const int crop_w = this->layer_param_.transform_param().crop_w();
//   bool force_color = this->layer_param_.data_param().force_encoded_color();
//   if (batch_size == 1 && crop_size == 0 && crop_h == 0 && crop_w == 0) {
//     Datum datum;
//     datum.ParseFromString(cursor_->value());
//     if (datum.encoded()) {
//       if (force_color) {
//         DecodeDatum(&datum, true);
//       } else {
//         DecodeDatumNative(&datum);
//       }
//     }
//     this->prefetch_data_.Reshape(1, datum.channels(),
//         datum.height(), datum.width());
//     this->transformed_data_.Reshape(1, datum.channels(),
//         datum.height(), datum.width());
//   }

//   Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
// >>>>>>> Non-square
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables

  if (this->output_labels_) {
    top_label = batch->label_.mutable_cpu_data();
  }
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    // get a datum
    Datum& datum = *(reader_.full().pop("Waiting for data"));
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply data transformations (mirror, scale, crop...)
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset);
    this->data_transformer_->Transform(datum, &(this->transformed_data_));
    // Copy label.
    if (this->output_labels_) {
      top_label[item_id] = datum.label();
    }
    trans_time += timer.MicroSeconds();

    reader_.free().push(const_cast<Datum*>(&datum));
  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(DataLayer);
REGISTER_LAYER_CLASS(Data);

}  // namespace caffe
