#include <windows.h>
#include <iostream>
#include <cstdlib>
#include <io.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
using namespace std;

void getFiles(string path, vector<string>& files);
void SplitString(const string& s, vector<string>& v, const string& c);
void getFileName(const string& filepath, string& name);