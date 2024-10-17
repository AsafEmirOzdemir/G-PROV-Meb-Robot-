#include <iostream>
#include <cmath>
#include <cstdlib>

using namespace std;

class NN_Math {
	public :
		NN_Math(int actIndexI) {
			actIndex = actIndexI;
		}

		void setEvent(int i) {
			actIndex = i;
		}

		long double act(long double i) {
			if(actIndex == 0) {
				return sigmoid(i);
			}
			else if(actIndex == 1) {
				return step(i);
			}
			else if(actIndex == 2) {
				return leakyRelu(i);
			}
			else if(actIndex == 3) {
				return tanh(i);
			}
			return -1;
		}

		long double dAct(long double i) {
			if(actIndex == 0) {
				return dSigmoid(i);
			}
			else if(actIndex == 1) {
				return dStep();
			}
			else if(actIndex == 2) {
				return dLeakyRelu(i);
			}
			else if(actIndex == 3) {
				return dTanh(i);
			}
			return -1;
		}

	private :
		int actIndex;
		long double leakyReluAlpha = 0.3;

		long double sigmoid(long double i) {
			return 1 / (1 + exp((float)-i));
		}
		long double step(long double i) {
			return i < 0 ? 0 : (i == 0 ? 0.5 : 1);
		}
		long double leakyRelu(long double i) {
			//return i > 0 ? i : leakyReluAlpha * i;
			return max(i * leakyReluAlpha, i);;
		}
		long double tanh(long double i) {
			return (exp(i) - exp(-i)) / (exp(i) + exp(-i));
		}

		long double dSigmoid(long double i) {
			return sigmoid(i) * (1 - sigmoid(i));
		}
		long double dStep() {
			return 0;
		}
		long double dLeakyRelu(long double i) {
			return i > 0 ? 1 : leakyReluAlpha;
		}
		long double dTanh(long double i) {
			return 1 - (tanh(i) * tanh(i));
		}
};

class NN {
	public :
		NN(int* layerMapI, int lengthOfNNI, long double** inputI, long double** outputI, int dataLengthI, long double minStartValues, long double maxStartValues) {
			lengthOfNN = lengthOfNNI;
			//cout << lengthOfNN - 1 << "a";
			inputLength = layerMapI[0];
			//cout << inputLength << "burada";
			outputLength = layerMapI[lengthOfNN - 1];
			dataLength = dataLengthI;
			layerMap = new int[lengthOfNN];
			equalArray(layerMap, layerMapI, lengthOfNN);
			lengthOfWeights = new int[lengthOfNN - 1];
			lengthOfNeurons = new int[lengthOfNN];
			for(int i = 0; i < lengthOfNN; i++) {
				lengthOfNeurons[i] = layerMap[i];
			}
			//cout << lengthOfNeurons[0] << "hh";
			weights = new long double**[lengthOfNN - 1];
			for(int i = 0; i < lengthOfNN - 1; i++) {
				weights[i] = new long double*[getMaxValue(layerMap, lengthOfNN)];
				for(int j = 0; j < getMaxValue(layerMap, lengthOfNN); j++) {
					weights[i][j] = new long double[getMaxValue(layerMap, lengthOfNN)];
				}
			}
			dWeights = new long double**[lengthOfNN - 1];
			for(int i = 0; i < lengthOfNN - 1; i++) {
				dWeights[i] = new long double*[getMaxValue(layerMap, lengthOfNN)];
				for(int j = 0; j < getMaxValue(layerMap, lengthOfNN); j++) {
					dWeights[i][j] = new long double[getMaxValue(layerMap, lengthOfNN)];
				}
			}
			for(int i = 0; i < lengthOfNN - 1; i++) {
				for(int j = 0; j < getMaxValue(layerMap, lengthOfNN); j++) {
					for(int k = 0; k < getMaxValue(layerMap, lengthOfNN); k++) {
						dWeights[i][j][k] = 0;
					}
				}
			}
			neurons = new long double*[lengthOfNN];
			for(int i = 0; i < lengthOfNN; i++) {
				neurons[i] = new long double[getMaxValue(layerMap, lengthOfNN)];
			}
			for(int i = 0; i < lengthOfNN; i++) {
				for(int j = 0; j < getMaxValue(layerMap, lengthOfNN); j++) {
					neurons[i][j] = 0;
				}
			}
			dNeurons = new long double*[lengthOfNN];
			for(int i = 0; i < lengthOfNN; i++) {
				dNeurons[i] = new long double[getMaxValue(layerMap, lengthOfNN)];
			}
			for(int i = 0; i < lengthOfNN; i++) {
				for(int j = 0; j < getMaxValue(layerMap, lengthOfNN); j++) {
					dNeurons[i][j] = 0;
				}
			}
			eNeurons = new long double*[lengthOfNN];
			for(int i = 0; i < lengthOfNN; i++) {
				eNeurons[i] = new long double[getMaxValue(layerMap, lengthOfNN)];
			}
			for(int i = 0; i < lengthOfNN; i++) {
				for(int j = 0; j < getMaxValue(layerMap, lengthOfNN); j++) {
					eNeurons[i][j] = 0;
				}
			}
			input = new long double*[dataLength];
			for(int i = 0; i < dataLength; i++) {
				input[i] = new long double[inputLength];
			}
			output = new long double*[dataLength];
			for(int i = 0; i < dataLength; i++) {
				output[i] = new long double[outputLength];
			}
			outputT = new long double*[dataLength];
			for(int i = 0; i < dataLength; i++) {
				outputT[i] = new long double[outputLength];
			}
			for(int i = 0; i < dataLength; i++) {
				for(int j = 0; j < inputLength; j++) {
					input[i][j] = inputI[i][j];
					//cout << inputI[i][1] << "e";
				}
				for(int j = 0; j < outputLength; j++) {
					output[i][j] = 0;
				}
				for(int j = 0; j < outputLength; j++) {
					outputT[i][j] = outputI[i][j];
				}
			}
			//cout << output[0][0];
			wBias = new long double*[lengthOfNN - 1];
			for(int i = 0; i < lengthOfNN - 1; i++) {
				wBias[i] = new long double[getMaxValue(layerMap, lengthOfNN)];
			}
			for(int i = 0; i < lengthOfNN - 1; i++) {
				for(int j = 0; j < getMaxValue(layerMap, lengthOfNN); j++) {
					wBias[i][j] = 0;
				}
			}
			dWBias = new long double*[lengthOfNN - 1];
			for(int i = 0; i < lengthOfNN; i++) {
				dWBias[i] = new long double[getMaxValue(layerMap, lengthOfNN)];
			}
			for(int i = 0; i < lengthOfNN - 1; i++) {
				for(int j = 0; j < getMaxValue(layerMap, lengthOfNN); j++) {
					dWBias[i][j] = 0;
				}
			}
			attachRandomValues(minStartValues, maxStartValues);
			//cout << "j" << calculate.act(5) << "g";
		}

		void run() {
			//cout << dataLength;
			for(int i = 0; i < dataLength; i++) {
				feedForward(i);
				backPropagation(i);
			}
		}

		void predict(long double* inputI, long double* outputI) {
			for(int j = 0; j < lengthOfNN; j++) {
				for(int k = 0; k < lengthOfNeurons[j]; k++) {
					neurons[j][k] = 0;
				}
			}
			for(int j = 0; j < inputLength; j++) {
				neurons[0][j] = inputI[j];
			}
			for(int j = 0; j < lengthOfNN - 1; j++) {
				for(int k = 0; k < lengthOfNeurons[j + 1]; k++) {
					for(int t = 0; t < lengthOfNeurons[j]; t++) {
						neurons[j + 1][k] += neurons[j][t] * weights[j][t][k];
					}
					neurons[j + 1][k] += wBias[j][k] * 1;
					neurons[j + 1][k] = calculate.act(neurons[j + 1][k]);
				}
			}
			for(int j = 0; j < outputLength; j++) {
				//cout << neurons[lengthOfNN - 1][0] << "hh";
				*(outputI + j) = neurons[lengthOfNN - 1][j];
			}
		}

	private :
		int* layerMap;
		int* lengthOfWeights;
		int* lengthOfNeurons;
		long double*** weights;
		long double** neurons;
		long double** input;
		long double** output;
		long double** outputT;
		int lengthOfNN;
		int inputLength;
		int outputLength;
		int dataLength;
		long double** dNeurons;
		long double** eNeurons;
		long double*** dWeights;
		NN_Math calculate = NN_Math(0);
		long double learningRate = 0.01;
		long double** wBias;
		long double** dWBias;

		void equalArray(int* target, int* key, int length) {
			for(int i = 0; i < length; i++) {
				*(target + i) = *(key + i);
			}
		}

		int getMaxValue(int* array, int length) {
			int biggest = *(array);
			for(int i = 1; i < length; i++) {
				biggest = *(array + i) > biggest ? *(array + i) : biggest;
			}
			return biggest;
		}

		void setNN(int i) {
			for(int j = 0; j < inputLength; j++) {
				//cout << input[i][j];
				neurons[0][j] = input[i][j];
				//cout << "b";
				//cout << input[i][1] << "j\n";
			}
		}

		void backPropagation(int i) {
			//cout << lengthOfNN - 1;
			setNN(i);
			for(int j = 0; j < lengthOfNN; j++) {
				for(int k = 0; k < lengthOfNeurons[j]; k++) {
					eNeurons[j][k] = 0;
					dNeurons[j][k] = 0;
				}
			}
			for(int j = lengthOfNN - 1; j > -1; j--) {
				for(int t = 0; t < lengthOfNeurons[j]; t++) {
					if(j == lengthOfNN - 1) {
						eNeurons[j][t] = outputT[i][t] - neurons[j][t];
						dNeurons[j][t] = eNeurons[j][t] * calculate.dAct(neurons[j][t]);
						//cout << dNeurons[i][t] << " ";
						//cout << output[i][t];
						//cout << dNeurons[j][t] << ",";
					}
					else {
						for(int k = 0; k < lengthOfNeurons[j + 1]; k++) {
							eNeurons[j][t] += weights[j][t][k] * dNeurons[j + 1][k];
							dWeights[j][t][k] = neurons[j][t] * dNeurons[j + 1][k];
							dWBias[j][k] = dNeurons[j + 1][k] * 1;
							weights[j][t][k] += dWeights[j][t][k] * learningRate;
							wBias[j][k] += dWBias[j][k] * learningRate;
							//cout << dWeights[j][t][k] << ",";
						}
						//cout << "\n";
						dNeurons[j][t] = eNeurons[j][t] * calculate.dAct(neurons[j][t]);
						//cout << dNeurons[j][t] << ", " << j << "\n";
					}
					//cout << "\n";
				}
			}
			//cout << neurons[0][1] << "\n";
			/*cout << dWeights[0][0][0] << " , " << dWeights[0][0][1] << "\n";
			cout << dWeights[0][1][0] << " , " << dWeights[0][1][1] << "\n";
			cout << dWeights[1][0][0] << " , " << dWeights[1][1][0] << "\n";
			cout << "-------------------------" << "\n";*/
			//cout << neurons[2][0] << "\n";
		}

		void feedForward(int i) {
			//cout << lengthOfNN;
			for(int j = 0; j < lengthOfNN; j++) {
				for(int k = 0; k < lengthOfNeurons[j]; k++) {
					neurons[j][k] = 0;
				}
			}
			setNN(i);
			for(int j = 0; j < lengthOfNN - 1; j++) {
				//cout << "a";
				for(int k = 0; k < lengthOfNeurons[j + 1]; k++) {
					for(int t = 0; t < lengthOfNeurons[j]; t++) {
						neurons[j + 1][k] += neurons[j][t] * weights[j][t][k];
						//cout << neurons[j + 1][k];
					}
					neurons[j + 1][k] += wBias[j][k] * 1;
					neurons[j + 1][k] = calculate.act(neurons[j + 1][k]);
				}
			}
			//cout << neurons[2][0] << "\n";
			//cout << calculate.act(neurons[0][1] * weights[0][1][0] + neurons[0][0] * weights[0][0][0]) << " = " << neurons[1][0] << "er\n";
			for(int j = 0; j < lengthOfNeurons[lengthOfNN - 1]; j++) {
				output[i][j] = neurons[lengthOfNN - 1][j];
			}
		}
		long double fRand(long double fMin, long double fMax)
		{
		    long double f = (long double)rand() / RAND_MAX;
		    return fMin + f * (fMax - fMin);
		}

		void attachRandomValues(long double b1, long double b2) {
			for(int i = 0; i < lengthOfNN - 1; i++) {
				for(int j = 0; j < getMaxValue(layerMap, lengthOfNN); j++) {
					for(int k = 0; k < getMaxValue(layerMap, lengthOfNN); k++) {
						wBias[i][k] = fRand(b1, b2) * (long double)sqrt(2 / lengthOfNeurons[i]);
						weights[i][j][k] = fRand(b1, b2) * (long double)sqrt(2 / lengthOfNeurons[i]);
						//cout << weights[i][j][k] << "\n";
					}
				}
			}
			/*cout << weights[0][0][0] << " , " << dWeights[0][0][1] << "\n";
			cout << weights[0][1][0] << " , " << dWeights[0][1][1] << "\n";
			cout << weights[1][0][0] << " , " << dWeights[1][1][0] << "\n";
			cout << "-------------------------" << "\n";*/
		}

};

long double fRand(long double fMin, long double fMax)
{
    long double f = (long double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

int main() {
	NN_Math n(2);
	cout << "oo" << n.dAct(-5) << "oo";
	long double b1 = -6;
	long double b2 = 2;
	//cout << ((fmod(rand(), abs(b1 - b2))) + min(b1, b2));
	long double* l = new long double[5];
	//cout << l[1];
	long double** input = new long double*[4000];
	long double** output = new long double*[4000];
	long double exampleCount = 4000;
	for(int i = 0; i < exampleCount; i++) {
		input[i] = new long double[2];
		output[i] = new long double[1];
	}
	for(int i = 0; i < exampleCount; i++) {
		long double b1 = 10;
		long double b2 = -10;
		input[i][0] = fRand(b1, b2);
		input[i][1] = fRand(b1, b2);
		output[i][0] = pow(input[i][0], 2) > input[i][1] + 6 ? 1 : -1;
	}
	//cout << input[0][0] << " " << input[0][1] << " " << output[0][0];
	int layerMap[3] = {2, 4, 1};
	NN nerualNetwork = NN(layerMap, 3, input, output, exampleCount, -3, 3);
	long double inputP[2] = {2, 0};
	long double outputP[1] = {0};
	for(int i = 0; i < 100; i++) {
		nerualNetwork.run();
		nerualNetwork.predict(inputP, outputP);
		cout << "result: " << outputP[0] << "\n";
	}
	if(outputP[0] < 0.5) {
		cout << "kucuk";
	}
	else {
		cout << "buyuk";
	}
	cin.get();
	return 0;
}
