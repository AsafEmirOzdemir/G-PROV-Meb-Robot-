#ifndef READDB_HPP
#define READDB_HPP
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
class DB{
public:
    int sizeOfValues = 0;
    int categoryInd = 0;
    const char* fileName;
    int lenghtofCategories = 1;
    std::vector<std::string> categories;
    std::vector<float> values;
    std::ifstream file;
    void saveCategoryvalues(const char *categoryName);
    void readCSV(const char *fileNames);
};
#endif
