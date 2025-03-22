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
#define LRate 0.8
#define MaxEpoches 1000
#define BatchSize 32
#define minLoss 0.5

namespace math
{
    double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
    double d_sigmoid(double x) { return sigmoid(x) * (1.0 - sigmoid(x)); }
    double softmax(std::vector<double> x, int index)
    {
        double deno{0};
        double maxElem = *std::max_element(x.begin(), x.end());
        double result;
        for (double i : x)
            deno += exp(i - maxElem);            // 这里减去最大值是为了数据稳定性
        result = exp(x[index] - maxElem) / deno; // 同理
        return result;
    }
    double CEL(std::vector<std::vector<double>> predics,std::vector<int> trueClasses)
    {
        double result{0};
        for (size_t i{0}; i < predics.size(); i++) result += -log(predics[i][trueClasses[i]]);
        return result / BatchSize;
    }
    std::vector<double> d_CEL(std::vector<std::vector<double>> predics, std::vector<int> trueClasses)
    {
        std::vector<double> result(predics[0].size(), 0.0);
        for (int i{0}; i < OutNes; i++){
            for(size_t j{0};j<BatchSize;j++){
                result[i] += (predics[j][i] - (i == trueClasses[j] ? 1.0 : 0.0))/BatchSize;
            }
        }
        // 这个计算公式是因为交叉熵和softmax结合后简化的
        return result;
    }

    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> wei(-(1.0 / sqrt(InNes)), (1.0 / sqrt(InNes)));
    std::uniform_real_distribution<double> b(-0.01, 0.01);
}

struct inl
{
    std::vector<double> ins;
    std::vector<std::vector<double>> weis;
    inl()
    {
        ins.resize(InNes);
        std::vector<double> temp;
        for (size_t i{0}; i < HidNes; i++)
        {
            for (size_t j{0}; j < InNes; j++)
                temp.push_back(math::wei(math::gen));
            weis.push_back(temp);
        }
    }
};
struct hidl
{
    std::vector<double> ins;
    std::vector<std::vector<double>> weis;
    std::vector<double> bs;
    hidl()
    {
        ins.resize(HidNes);
        std::vector<double> temp;
        for (size_t i{0}; i < OutNes; i++)
        {
            bs.push_back(math::b(math::gen));;
            for (size_t j{0}; j < HidNes; j++)
                temp.push_back(math::wei(math::gen));
            weis.push_back(temp);
        }
    }
    std::vector<std::vector<double>> weisTranspose()
    {
        std::vector<std::vector<double>> result = std::vector<std::vector<double>>(HidNes);
        for (size_t i{0}; i < HidNes; i++)
            result[i] = std::vector<double>(OutNes);
        for (size_t i{0}; i < OutNes; i++)
        {
            for (size_t j{0}; j < HidNes; j++)
                result[j][i] = weis[i][j];
        }
        return result;
    }
};
struct outl
{
    std::vector<double> ins;
    std::vector<double> bs;
    outl()
    {
        ins.resize(OutNes);
        for (size_t i{0}; i < OutNes; i++)
            bs.push_back(math::b(math::gen));
    }
};

struct sample
{
    unsigned char trData[InNes];
    unsigned char label;
};
std::vector<sample> loadData(const std::string trName, const std::string lbName)
{
    std::vector<sample> result;
    sample temp;
    std::ifstream tr(trName, std::ios::binary);
    std::ifstream lb(lbName, std::ios::binary);
    if (tr.is_open() && lb.is_open())
    {
        // Skip the header
        tr.ignore(16);
        lb.ignore(8);

        unsigned char trTmp, laTmp;
        
        for(size_t i{0};tr.read(reinterpret_cast<char *>(&trTmp), sizeof(trTmp)) && lb.read(reinterpret_cast<char*>(&laTmp), sizeof(laTmp));i++){
            temp.trData[i] = trTmp;
            if (sizeof(temp.trData) == 784)
            {
                temp.label = laTmp;
                result.push_back(temp);
                for(size_t j{0};j<784;j++) temp.trData[j] = 0;
            }
            i = 0;
        }
    }
    else
    {
        std::cout << "Error occurs opening files." << std::endl;
        exit(-1);
    }
    tr.close();
    lb.close();
    return result;
}


int main()
{
    std::vector<sample> trData = loadData("emnist-mnist-train-images-idx3-ubyte","emnist-mnist-train-labels-idx1-ubyte");
    std::vector<sample> teData = loadData("emnist-mnist-test-images-idx3-ubyte","emnist-mnist-test-labels-idx1-ubyte");

    int epoch{0}, batch{0};
    double loss{0};
    inl in;
    hidl hid;
    outl out;
    while (epoch < MaxEpoches)
    {
        // forward
        std::vector<std::vector<double>> predics;
        std::vector<double> rawPridictions;
        std::vector<int> trueClasses;
        for (; batch < BatchSize; batch++)
        {
            // Input layer to hidden layer
            for (size_t i{0}; i < InNes; i++)
                in.ins[i] = trData[batch].trData[i];
            for (size_t i{0}; i < HidNes; i++)
            {
                for (size_t j{0}; j < InNes; j++)
                    hid.ins[i] += in.ins[i] * in.weis[i][j];
            }
            // Hidden layer to output layer
            for (size_t i{0}; i < OutNes; i++)
            {
                double sum{0};
                for (size_t j{0}; j < HidNes; j++)
                {
                    sum += math::sigmoid(hid.ins[j] + hid.bs[j]) * hid.weis[i][j];
                    // hidOutputs.push_back(math::sigmoid(hid.ins[j] + hid.bs[j]));
                }
                out.ins[i] = sum + out.bs[i];
            }
            // Output layer to predictions
            for(size_t i{0};i<OutNes;i++) rawPridictions.push_back(math::softmax(out.ins, trData[i].label));
            predics.push_back(rawPridictions);
            rawPridictions.clear();
            trueClasses.push_back(trData[batch].label);
        }
        // backward
        loss = math::CEL(predics,trueClasses);
        if (loss < minLoss)
            break;
        else
        {
            // update output layer
            std::vector<double> delta4out = math::d_CEL(predics, trueClasses);
            for (size_t i{0}; i < delta4out.size(); i++)
                delta4out[i] *= LRate;
            for (size_t i{0}; i < OutNes; i++)
                out.bs[i] -= delta4out[i];

            // update hidden layer
            std::vector<double> delta4hidden(HidNes,0.0);
            std::vector<std::vector<double>> transposedWeis = hid.weisTranspose();
            for (size_t i{0}; i < HidNes; i++)
            {
                for (size_t j{0}; j < OutNes; j++)
                {
                    delta4hidden[i] += transposedWeis[i][j] * delta4out[j];
                }
            }
            for (size_t i{0}; i < HidNes; i++)
                delta4hidden[i] *= LRate * math::d_sigmoid(hid.ins[i]);
            for (size_t i{0}; i < OutNes; i++)
            {
                for (size_t j{0}; j < HidNes; j++)
                    hid.weis[i][j] -= delta4hidden[j] * hid.ins[j];
            }
            for (size_t i{0}; i < HidNes; i++)
                hid.bs[i] -= delta4hidden[i];

            // update input layer
            std::vector<std::vector<double>> delta4input(HidNes, std::vector<double>(InNes,0.0));
            for (size_t i = 0; i < HidNes; i++)
            {
                for (size_t j = 0; j < InNes; j++)
                {
                    delta4input[i][j] += delta4hidden[i] * in.weis[i][j]*math::d_sigmoid(in.ins[j]);
                }
            }

            for (size_t i = 0; i < HidNes; i++)
            {
                for (size_t j = 0; j < InNes; j++)
                {
                    in.weis[i][j] -= LRate * delta4input[i][j] * in.ins[j];
                }
            }
        }
        std::cout<<"Current epoch: "<<epoch<<" Loss: "<<loss<<std::endl;
        epoch++;
    }
    return 0;
}
