#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <memory>
#include <fstream>
#include <algorithm>

#define InNes 28 * 28
#define HidNes 100
#define OutNes 10
#define LRate 0.01 // 降低学习率以提高稳定性
#define MaxEpoches 100
#define BatchSize 32
#define minLoss 0.1 // 调整目标损失

namespace math
{
    inline double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
    inline double d_sigmoid(double x) { return x * (1.0 - x); } // 输入应为激活后的值
    inline void softmax(std::vector<double> &x)
    {
        double maxElem = *std::max_element(x.begin(), x.end());
        double deno = 0.0;
        for (auto &v : x)
            deno += exp(v - maxElem);
        for (auto &v : x)
            v = exp(v - maxElem) / deno;
    }

    double CEL(const std::vector<std::vector<double>> &predics, const std::vector<uint8_t> &trueClasses)
    {
        double result{0};
        for (size_t i = 0; i < predics.size(); ++i)
            result += -log(predics[i][trueClasses[i]]);
        return result / predics.size();
    }

    std::mt19937 gen(std::random_device{}());
    std::normal_distribution<double> wei(0.0, 1.0 / sqrt(InNes)); // 使用正态分布初始化
    std::uniform_real_distribution<double> b(-0.1, 0.1);
}

struct Network
{
    struct Layer
    {
        std::vector<std::vector<double>> weights;
        std::vector<double> biases;

        Layer(size_t input_size, size_t output_size) : weights(output_size, std::vector<double>(input_size)),
                                                       biases(output_size)
        {
            initialize();
        }

        void initialize()
        {
            for (auto &row : weights)
                for (auto &w : row)
                    w = math::wei(math::gen);
            for (auto &b : biases)
                b = math::b(math::gen);
        }
    };

    Layer input_hidden;
    Layer hidden_output;

    Network() : input_hidden(InNes, HidNes), hidden_output(HidNes, OutNes) {}

    std::vector<double> forward(const std::vector<uint8_t> &input)
    {
        std::vector<double> hidden(HidNes);

        // Input to hidden
        for (size_t i = 0; i < HidNes; ++i)
        {
            double sum = input_hidden.biases[i];
            for (size_t j = 0; j < InNes; ++j)
                sum += input[j] * input_hidden.weights[i][j];
            hidden[i] = math::sigmoid(sum);
        }

        // Hidden to output
        std::vector<double> output(OutNes);
        for (size_t i = 0; i < OutNes; ++i)
        {
            output[i] = hidden_output.biases[i];
            for (size_t j = 0; j < HidNes; ++j)
                output[i] += hidden[j] * hidden_output.weights[i][j];
        }

        math::softmax(output);
        return output;
    }

    void backward(const std::vector<std::vector<uint8_t>> &batch_inputs,
                  const std::vector<uint8_t> &batch_labels)
    {
        // 存储中间结果
        std::vector<std::vector<double>> hidden_activations;
        std::vector<std::vector<double>> outputs;

        // 前向传播并保存中间结果
        for (const auto &input : batch_inputs)
        {
            std::vector<double> hidden(HidNes);
            for (size_t i = 0; i < HidNes; ++i)
            {
                double sum = input_hidden.biases[i];
                for (size_t j = 0; j < InNes; ++j)
                    sum += input[j] * input_hidden.weights[i][j];
                hidden[i] = math::sigmoid(sum);
            }
            hidden_activations.push_back(hidden);

            std::vector<double> output(OutNes);
            for (size_t i = 0; i < OutNes; ++i)
            {
                output[i] = hidden_output.biases[i];
                for (size_t j = 0; j < HidNes; ++j)
                    output[i] += hidden[j] * hidden_output.weights[i][j];
            }
            math::softmax(output);
            outputs.push_back(output);
        }

        // 计算输出层梯度
        std::vector<double> delta_output(OutNes, 0.0);
        for (size_t i = 0; i < OutNes; ++i)
        {
            for (size_t j = 0; j < batch_inputs.size(); ++j)
            {
                delta_output[i] += (outputs[j][i] - (i == batch_labels[j] ? 1.0 : 0.0));
            }
            delta_output[i] /= batch_inputs.size();
        }

        // 更新输出层参数
        for (size_t i = 0; i < OutNes; ++i)
        {
            hidden_output.biases[i] -= LRate * delta_output[i];
            for (size_t j = 0; j < HidNes; ++j)
            {
                double grad = 0.0;
                for (size_t k = 0; k < batch_inputs.size(); ++k)
                    grad += hidden_activations[k][j] * delta_output[i];
                hidden_output.weights[i][j] -= LRate * grad / batch_inputs.size();
            }
        }

        // 计算隐藏层梯度
        std::vector<double> delta_hidden(HidNes, 0.0);
        for (size_t j = 0; j < HidNes; ++j)
        {
            for (size_t i = 0; i < OutNes; ++i)
            {
                delta_hidden[j] += hidden_output.weights[i][j] * delta_output[i];
            }
            // 计算平均梯度并应用激活导数
            double avg_act = 0.0;
            for (size_t k = 0; k < batch_inputs.size(); ++k)
                avg_act += hidden_activations[k][j];
            avg_act /= batch_inputs.size();
            delta_hidden[j] *= math::d_sigmoid(avg_act);
        }

        // 更新隐藏层参数
        for (size_t i = 0; i < HidNes; ++i)
        {
            input_hidden.biases[i] -= LRate * delta_hidden[i] / batch_inputs.size();
            for (size_t j = 0; j < InNes; ++j)
            {
                double grad = 0.0;
                for (size_t k = 0; k < batch_inputs.size(); ++k)
                    grad += batch_inputs[k][j] * delta_hidden[i];
                input_hidden.weights[i][j] -= LRate * grad / batch_inputs.size();
            }
        }
    }
};

struct Sample
{
    std::vector<uint8_t> pixels;
    uint8_t label;
};

std::vector<Sample> load_dataset(const std::string &image_path, const std::string &label_path)
{
    std::vector<Sample> dataset;
    std::ifstream images(image_path, std::ios::binary);
    std::ifstream labels(label_path, std::ios::binary);

    if (!images || !labels)
    {
        std::cerr << "Error opening files!" << std::endl;
        return dataset;
    }

    // 跳过头部信息
    images.seekg(16);
    labels.seekg(8);

    while (true)
    {
        Sample s;
        s.pixels.resize(InNes);
        if (!images.read(reinterpret_cast<char *>(s.pixels.data()), InNes))
            break;
        if (!labels.read(reinterpret_cast<char *>(&s.label), 1))
            break;
        dataset.push_back(s);
    }

    return dataset;
}

int main()
{
    auto train_data = load_dataset("emnist-mnist-train-images-idx3-ubyte",
                                   "emnist-mnist-train-labels-idx1-ubyte");
    auto test_data = load_dataset("emnist-mnist-test-images-idx3-ubyte",
                                  "emnist-mnist-test-labels-idx1-ubyte");

    Network network;

    int epoch{0};
    while(epoch<MaxEpoches)
    {
        double total_loss = 0.0;
        int correct = 0;

        // 打乱训练数据
        std::shuffle(train_data.begin(), train_data.end(), math::gen);

        for (size_t i = 0; i < train_data.size(); i += BatchSize)
        {
            size_t end = std::min(i + BatchSize, train_data.size());
            size_t actual_batch_size = end - i;

            std::vector<std::vector<uint8_t>> batch_inputs;
            std::vector<uint8_t> batch_labels;

            // 准备批次数据
            for (size_t j = i; j < end; ++j)
            {
                batch_inputs.push_back(train_data[j].pixels);
                batch_labels.push_back(train_data[j].label);
            }

            // 前向传播
            std::vector<std::vector<double>> outputs;
            for (const auto &input : batch_inputs)
                outputs.push_back(network.forward(input));

            // 计算损失和准确率
            total_loss += math::CEL(outputs, batch_labels);
            for (size_t j = 0; j < outputs.size(); ++j)
            {
                auto pred = std::max_element(outputs[j].begin(), outputs[j].end());
                if (std::distance(outputs[j].begin(), pred) == batch_labels[j])
                    ++correct;
            }

            // 反向传播
            network.backward(batch_inputs, batch_labels);
        }

        // 输出训练信息
        double avg_loss = total_loss / (train_data.size() / BatchSize);
        double accuracy = static_cast<double>(correct) / train_data.size();
        std::cout << "Epoch " << epoch + 1
                  << " Loss: " << avg_loss
                  << " Accuracy: " << accuracy * 100 << "%" << std::endl;

        if (avg_loss < minLoss)
            break;
        epoch++;
        _sleep(10000);
    }

    // 测试代码（可选）
    // ...

    return 0;
}