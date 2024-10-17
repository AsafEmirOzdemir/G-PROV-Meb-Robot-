#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "Libs/ReadDB.hpp"
using namespace std;
void DB::readCSV(const char* fileNames){
    lenghtofCategories = 0;
    categories = {};
    fileName = fileNames;
    file.open(fileName);
    string types[0];
    string fullLine;
    getline(file, fullLine);
    string flood = "";
    for (int i = 0; i < (int)(fullLine.length()); i++){
        if(fullLine[i] != ','){
            flood = flood + fullLine[i];
        }
        else{
            categories.push_back(flood);
            flood = "";
            lenghtofCategories++;
        }
    }
    categories.push_back(flood);
    flood = "";
    file.close();
}
void DB::saveCategoryvalues(const char* categoryName){
    values = {};
    sizeOfValues = 0;
    categoryInd = 0;
    std::string tmp;
    int ind = 0;
    std::string tmpNUM = "";
    for (int i = 0; i < lenghtofCategories; i++){
        if(categories[i] != categoryName){
            categoryInd++;
        }
        else{
            break;
        }
    }
    file.open(fileName);
    getline(file, tmp);
    while (getline(file, tmp))
    {
        for (int i = 0; i < (int)(tmp.length()); i++){
            if(categoryInd == ind){
                if(tmp[i] != ','){
                    tmpNUM = tmpNUM + tmp[i];
                }
            }
            if(tmp[i] == ','){
                ind++;
            }
            if(categoryInd < ind){
                break;
            }
        }
        values.push_back(stof(tmpNUM));
        sizeOfValues++;
        tmpNUM = "";
        ind = 0;
    }
    file.close();
}
