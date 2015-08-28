// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
void ZeroOutputLayer<Dtype>::SetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  Layer<Dtype>::SetUp(bottom, top);
  (*top)[0]->Reshape(1, 1, 1, 1);
}

template <typename Dtype>
Dtype ZeroOutputLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  // LOG(INFO) << "Accuracy: " << accuracy;
  (*top)[0]->mutable_cpu_data()[0] = Dtype(0);

  // Accuracy layer should not be used as a loss function.
  return Dtype(0);
}

INSTANTIATE_CLASS(ZeroOutputLayer);

}  // namespace caffe
