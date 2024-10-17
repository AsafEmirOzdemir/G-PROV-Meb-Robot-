#ifndef HayatNN_HPP
#define HayatNN_HPP

class NN{
    private:
    long double **input;
    long double **output;
    long double **outputT;
    long double ***weights;
    long double ***dWeights;
    long double **neurons;
    long double **dNeurons;
    long double **eNeurons;
    long double **wBias;
    long double **dwBias;
    int* connectionMap;
    int* layerMap;
    int ACT = 3;
    int lenghtofNN;
    int lenghtofConnections;
    float leakyReluAlpha = 0.3f;
    long double learningRate = 0.001;
    void CopyArray(int* toThisArray, int* fromThisArray, int arraylenght);
    double GetRandValue(double min, double max);
    long double act(long double i);
    long double dAct(long double i);
    public:
    NN(int* NNMap, int lenghtofNNMap, int* connectionsMap, int connectionsMapLenght);
    void Learn(long double **Input, int dataLenght, long double **Output, int epoch);
    void Predict(long double *Input, long double *Output);
    void SetLearningRate(long double learnRate);
    void SetACT(int actIndex);
};

#endif