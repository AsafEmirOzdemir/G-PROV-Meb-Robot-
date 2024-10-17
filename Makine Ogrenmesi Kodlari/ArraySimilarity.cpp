#include <cmath>
#include "Libs/ArraySimilarity.hpp"

float ArraySIM::getSIM(float* a, float* b, int lengthOfArrays) {
	float x = 0;
	float y = 0;
	float z = 0;
	for(int i = 0; i < lengthOfArrays; i++) {
		x += *(a + i) * *(b + i);
		y += *(a + i) * *(a + i);
		z += *(b + i) * *(b + i);
	}
	return 100 * x / (sqrt(y) * sqrt(z));
}
