#include <iostream>
#include "NN.hpp"
#include <cstdlib>
#include <cmath>
#include <fstream>

using namespace std;

NN::NN(int* NNMap, int lenghtofNNMap, int* connectionsMap, int connectionsMapLenght){
    layerMap = new int[lenghtofNNMap];
    lenghtofNN = lenghtofNNMap;
    connectionMap = new int[connectionsMapLenght + ((connectionsMapLenght/3)*2)];
    CopyArray(layerMap, NNMap, lenghtofNNMap);
    weights = new long double**[lenghtofNNMap-1];
    for(int i = 0; i < lenghtofNNMap-1; i++){
        weights[i] = new long double*[layerMap[i]];
        for(int j = 0; j < layerMap[i]; j++){
            weights[i][j] = new long double[layerMap[1+i]];
        }
    }
    dWeights = new long double**[lenghtofNNMap-1];
    for(int i = 0; i < lenghtofNNMap-1; i++){
        dWeights[i] = new long double*[layerMap[i]];
        for(int j= 0; j < layerMap[i]; j++){
            dWeights[i][j] = new long double[layerMap[i+1]];
            for(int k = 0; k < layerMap[i+1]; k++){
                dWeights[i][j][k] =0;
            }
        }
    }
    neurons = new long double*[lenghtofNNMap];
    for(int i = 0; i < lenghtofNNMap; i++){
        neurons[i] = new long double[layerMap[i]];
        for(int j = 0; j < layerMap[i]; j++){
            neurons[i][j] = 0;
        }
    }
    dNeurons = new long double*[lenghtofNNMap];
    for(int i = 0; i < lenghtofNNMap; i++){
        dNeurons[i] = new long double[layerMap[i]];
        for (int j = 0; j < layerMap[i]; j++){
            dNeurons[i][j] = 0;
        }
    }
    eNeurons = new long double*[lenghtofNNMap];
    for(int i = 0; i < lenghtofNNMap; i++){
        eNeurons[i] = new long double[layerMap[i]];
        for (int j = 0; j < layerMap[i]; j++){
            eNeurons[i][j] = 0;
        }
    }
    wBias = new long double*[lenghtofNNMap-1];
    for(int i = 0; i < lenghtofNNMap-1; i++){
        wBias[i] = new long double[layerMap[i+1]];
        for(int j = 0; j < layerMap[i+1]; j++){
            wBias[i][j] = 0;
        }
    }
    dwBias = new long double*[lenghtofNNMap-1];
    for(int i = 0; i < lenghtofNNMap-1; i++){
        dwBias[i] = new long double[layerMap[i+1]];
        for(int j = 0; j < layerMap[i+1]; j++){
            dwBias[i][j] = 0;
        }
    }
    int a = 0;
    for(int i = 0; i < connectionsMapLenght; i+=3){
        int locationofnode,locationofnode0;
        int tmp = 0;
        int layerposition = 0;
        for(int j = 0; j < lenghtofNNMap; j++){
            tmp += layerMap[j];
            if(tmp >= connectionsMap[i]){
                locationofnode = (connectionsMap[i] - (tmp-layerMap[j]))-1;
                locationofnode0 = (connectionsMap[i+1] - (tmp-layerMap[j]))-1;
                break;
            }
            else{layerposition++;}
        }
        connectionMap[a] = layerposition;
        connectionMap[a+1] = locationofnode;
        connectionMap[a+2] = locationofnode0;

        int locationofnode1;
        tmp = 0;
        layerposition = 0;
        for(int j = 0; j < lenghtofNNMap; j++){
            tmp += layerMap[j];
            if(tmp >= connectionsMap[i+2]){
                locationofnode1 = (connectionsMap[i+2] - (tmp-layerMap[j]))-1;
                break;
            }
            else{layerposition++;}
        }
        connectionMap[a+3] = layerposition;
        connectionMap[a+4] = locationofnode1;
        a+=5;
    }
    lenghtofConnections = connectionsMapLenght + (connectionsMapLenght/3)*2;
    for(int i = 0; i < lenghtofConnections; i+=5){
        wBias[connectionMap[i+3]-1][connectionMap[i+4]] = GetRandValue(-2.0, 2.0) * (long double)sqrt(2 / layerMap[connectionMap[i]]);
        for(int j = connectionMap[i+1]; j <= connectionMap[i+2]; j++){
            weights[connectionMap[i]][j][connectionMap[i+4]] = GetRandValue(-2.0,2.0) * (long double)sqrt(2 / layerMap[connectionMap[i]]);
        }
    }
}
void NN::Learn(long double **Input, int dataLenght, long double **Output, int epoch){
    //Create Inputs
    input = new long double*[dataLenght];
    for(int i = 0; i < dataLenght; i++){
        input[i] = new long double[layerMap[0]];
        for(int j = 0; j < layerMap[0]; j++){
            input[i][j] = Input[i][j];
        }
    }
    //Create Real Outputs
    outputT = new long double*[dataLenght];
    for(int i = 0; i < dataLenght; i++){
        outputT[i] = new long double[layerMap[lenghtofNN-1]];
        for(int j = 0; j < layerMap[0]; j++){
            outputT[i][j] = Output[i][j];
        }
    }

    //feedforward
    for(int g = 0; g < epoch; g++){
        for(int i = 0; i < dataLenght; i++){
            //Reset neurons
            for(int j = 0; j < lenghtofNN; j++){
                for(int k = 0; k < layerMap[j]; k++){
                    neurons[j][k] = 0;
                }
            }
            //Save Inputs
            for(int j = 0; j < layerMap[0]; j++){
                neurons[0][j] = input[i][j];
            }
            //Mutiply with weights and save to neurons
            for(int j = 1; j < lenghtofNN; j++){
                for(int k = 0; k < layerMap[j]; k++){
                    for(int l = 0; l < lenghtofConnections; l+=5){
                        if(connectionMap[l+3] == j && connectionMap[l+4] == k){
                            for(int m = connectionMap[l+1]; m <= connectionMap[l+2]; m++){
                                neurons[j][k] += neurons[connectionMap[l]][m] * weights[connectionMap[l]][m][k];
                            }
                        }
                    }
                    neurons[j][k] += wBias[j-1][k] * 1;
                    neurons[j][k] = act(neurons[j][k]);
                }
            }
        //backPropagation
            //eNeurons'u ve dNeurons'u sıfırla.
            for(int j = 0; j < lenghtofNN; j ++){
                for(int k = 0; k < layerMap[j]; k++){
                    eNeurons[j][k] = 0;
                    dNeurons[j][k] = 0;
                }
            }
            //Katmanlar için backPropagation yap.
            for(int j = lenghtofNN-1; j > -1; j--){
                if(j == lenghtofNN -1){
                    for(int k = 0; k < layerMap[j]; k++){
                        eNeurons[j][k] = outputT[i][k] - neurons[j][k];
                        dNeurons[j][k] = eNeurons[j][k] * dAct(neurons[j][k]);
                        dwBias[j-1][k] = 1 * dNeurons[j][k];
                        wBias[j-1][k] += dwBias[j-1][k] * learningRate;
                    }
                }
                else{
                    for (int k = 0; k < layerMap[j]; k++){
                        for(int l = 0; l < lenghtofConnections; l+=5){
                            if(connectionMap[l] == j && connectionMap[l+1] <= k && connectionMap[l+2] >= k){
                                eNeurons[j][k] += weights[j][k][connectionMap[l+4]] * dNeurons[connectionMap[l+3]][connectionMap[l+4]];
                                dWeights[j][k][connectionMap[l+4]] = neurons[j][k] * dNeurons[connectionMap[l+3]][connectionMap[l+4]];
                                weights[j][k][connectionMap[l+4]] += dWeights[j][k][connectionMap[l+4]] * learningRate;
                            }
                        }
                        dNeurons[j][k] = eNeurons[j][k] * dAct(neurons[j][k]);
                        if(j != 0){
                            dwBias[j-1][k] = 1 * dNeurons[j][k];
                            wBias[j-1][k] += dwBias[j-1][k] * learningRate;
                        }
                    }
                }
            }
        }
    }
}
void NN::Predict(long double *Input, long double *Output){
    for(int i = 0; i < lenghtofNN; i++){
        for(int j = 0; j  < layerMap[i]; j++){
            neurons[i][j] = 0;
        }
    }
    for(int i = 0; i < layerMap[0]; i++){
        neurons[0][i] = Input[i];
    }
    for(int j = 1; j < lenghtofNN; j++){
        for(int k = 0; k < layerMap[j]; k++){
            for(int l = 0; l < lenghtofConnections; l+=5){
                if(connectionMap[l+3] == j && connectionMap[l+4] == k){
                    for(int m = connectionMap[l+1]; m <= connectionMap[l+2]; m++){
                        neurons[j][k] += neurons[connectionMap[l]][m] * weights[connectionMap[l]][m][k];
                        cout << connectionMap[l] << " - " << m << " - " << k << " : " << weights[connectionMap[l]][m][k] << endl;
                    }
                }
            }
            neurons[j][k] += wBias[j-1][k] * 1;
            cout << "Bias : " << j << " - " << k << " : " << wBias[j-1][k] << endl;
            neurons[j][k] = act(neurons[j][k]);
        }
    }
    for(int i = 0; i < layerMap[lenghtofNN-1]; i++){
        *(Output +i) = neurons[lenghtofNN-1][i];
    }
}

void NN::CopyArray(int* toThisArray, int* fromThisArray, int arraylenght){
    for(int i = 0; i < arraylenght; i++){
        *(toThisArray +i) = *(fromThisArray +i);
    }
}

double NN::GetRandValue(double min, double max){
    long double f = (long double)rand() / RAND_MAX;
    return min + f * (max - min);
}

void NN::SetLearningRate(long double learnRate){
    learningRate = learnRate;
}
void NN::SetACT(int actIndex){
    ACT = actIndex;
}
long double NN::act(long double i){
    switch (ACT)
    {
    case 0:
        return 1/(1+exp((float)-i));
    case 1:
        return i < 0 ? 0: (i==0?0.5:1);
    case 2:
        return max(i * (long double)leakyReluAlpha, i);
    case 3:
        return (exp(i)-exp(-i)) / (exp(i)+exp(-i));
    }
}
long double NN::dAct(long double i){
    switch (ACT)
    {
    case 0:
        return (1/(1+exp((float)-i))) * (1-(1/(1+exp((float)-i))));
    case 1:
        return 0;
    case 2:
        return 1>0?1:(long double)leakyReluAlpha;
    case 3:
        return 1 - (((exp(i)-exp(-i)) / (exp(i)+exp(-i))) * ((exp(i)-exp(-i)) / (exp(i)+exp(-i))));
    }
}