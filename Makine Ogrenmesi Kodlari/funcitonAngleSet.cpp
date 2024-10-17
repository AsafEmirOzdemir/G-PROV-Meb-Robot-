#include <iostream>
#include <cmath>

using namespace std;

double getAngleFromEdge(double target, double a0, double a1);
double getHipo(double* p0, double* p1);

int main() {
	cout << (acos(cos((90 * M_PI) / 180)) * 180) / M_PI << "\n";
	int widePointsLength = 9;
	double pointsWide[9][2] = {{0, 8}, {0, 4}, {0, 0}, {0, -4}, {4, -7}, {5, -11}, {0, -12}, {0, -14}, {0, -16}};
	double** points;
	int startPoint = -1;
	int endPoint = -1;
	double plotBias = 0.1;
	for(int i = 0; i < widePointsLength - 2; i++) {
		double plot0 = abs(pointsWide[i][0] - pointsWide[i][0]) == 0 ? 0 : abs(pointsWide[i][1] - pointsWide[i + 1][1]) / abs(pointsWide[i][0] - pointsWide[i][0]);
		double plot1 = abs(pointsWide[i + 2][0] - pointsWide[i + 1][0]) == 0 ? 0 : abs(pointsWide[i + 1][1] - pointsWide[i + 2][1]) / abs(pointsWide[i + 2][0] - pointsWide[i + 1][0]);
		if(startPoint == -1 && abs(plot0 - plot1) > plotBias) {
			startPoint = i;
		}
		else if(startPoint != -1 && abs(plot0 - plot1) < plotBias) {
			endPoint = i;
		}
	}
	points = new double*[abs(endPoint - startPoint) + 2];
	for(int i = startPoint; i < endPoint + 2; i++) {
		points[i - startPoint] = new double[2];
		points[i - startPoint][0] = pointsWide[i][0];
		points[i - startPoint][1] = pointsWide[i][1];
	}
	for(int i = 0; i < endPoint - startPoint + 2; i++) {
		cout << points[i][0] << " , " << points[i][1] << "\n";
	}
	int pointsLength = 6;
	//double points[6][2] = {{0, 0}, {0, -4}, {4, -7}, {5, -11}, {0, -12}, {0, -14}};
	double lineLength = getHipo(points[0], points[1]);
	double newPoint[2] = {lineLength, 0};
	double otherLine = getHipo(points[1], newPoint);
	double alpha = getAngleFromEdge(otherLine, lineLength, lineLength);
	double pointsNew[pointsLength - 2][2];

	for(int i = 1; i < pointsLength - 1; i++) {
		pointsNew[i - 1][0] = points[i][0] * cos((alpha * M_PI) / 180) - points[i][1] * sin((alpha * M_PI) / 180);
		pointsNew[i - 1][1] = points[i][0] * sin((alpha * M_PI) / 180) + points[i][1] * cos((alpha * M_PI) / 180);
	}
	double xDiff = pointsNew[0][0];
	double yDiff = pointsNew[0][1];
	for(int i = 0; i < pointsLength -2; i++) {
		pointsNew[i][0] -= xDiff;
		pointsNew[i][1] -= yDiff;
	}

	double xDistanceArea = pointsNew[pointsLength - 3][0] - pointsNew[0][0];
	double yDistanceArea = abs(pointsNew[pointsLength - 3][1] - pointsNew[0][1]);
	double edges[pointsLength - 2];

	for(int i = 0; i < pointsLength - 2; i++) {
		edges[i] = abs(pointsNew[i][1] - (abs(pointsNew[0][0] - pointsNew[i][0]) / xDistanceArea) * yDistanceArea);
	}

	double area = 0;
	for(int i = 0; i < pointsLength - 3; i++) {
		area += ((edges[i] + edges[i + 1]) * abs(pointsNew[i][0] - pointsNew[i + 1][0])) / 2;
	}
	cout << "----------------------------\n";
	cout << pointsNew[0][0] << " , " << pointsNew[0][1] << "\n";
	cout << pointsNew[1][0] << " , " << pointsNew[1][1] << "\n";
	cout << pointsNew[2][0] << " , " << pointsNew[2][1] << "\n";
	cout << pointsNew[3][0] << " , " << pointsNew[3][1] << "\n";
	cout << "\n" << area;
	cin.get();
	return 0;
}

double getAngleFromEdge(double target, double a0, double a1) {
	return (acos(((a0 * a0) + (a1 * a1) - (target * target)) / (2 * a0 * a1)) * 180) / M_PI;
}

double getHipo(double* p0, double* p1) {
	return sqrt((abs(p0[0] - p1[0]) * abs(p0[0] - p1[0])) + ((p1[1] - p0[1]) * (p1[1] - p0[1])));
}
