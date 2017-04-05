#include <glob.h>
#include <libgen.h>
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <cassert>
#include <string.h>
#include <sstream>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <cmath>
#include <stack>
#include <utility>
#include <queue>
#include <cfloat>
#include <numeric>

using namespace std;

static bool g_h;

//Glob function
inline std::vector<std::string> glob(const std::string& pat, bool sample){
    using namespace std;
    glob_t glob_result;
    glob(pat.c_str(),GLOB_TILDE,NULL,&glob_result);
    vector<string> ret;
    for(unsigned int i=0;i<glob_result.gl_pathc;++i){
        string str = string(glob_result.gl_pathv[i]);
        if (!sample) {
            if (g_h) {
                ret.push_back(str.substr(str.size()-15,15));
            }
            else {
                ret.push_back(str.substr(str.size()-14,14));
            }
            
        }
        else {
                ret.push_back(str.substr(str.size()-13,13));
        }
        
    }
    globfree(&glob_result);
    return ret;
}

class PreProcess{

public:
    vector<vector<bool>> train_mat;
    vector<vector<int>> Nc;
    double N;
    vector<vector<bool>> test_mat;
    vector<string> test_filenames;
    
    PreProcess(string directory, vector<string>& vocab, unordered_set<int>& all_labels){
        
        vector<string> train_filenames = glob(directory + "*train*", 0);
        N = static_cast<double>(train_filenames.size());
        train_mat.resize(N, vector<bool>(int(vocab.size()),0));
        int nc_idx = 0;
        Nc.resize(1);
        
        //update all_labels and Nc
        for (int i = 0; i < N; ++i) {
            
            std::pair<unordered_set<int>::iterator, bool> _pair = all_labels.insert((train_filenames[i][6] - '0')*10 + (train_filenames[i][7] - '0'));
            
            if (*(_pair.first) == (nc_idx)) {
                Nc[nc_idx].push_back(i);
            }
            else {
                nc_idx++;
                Nc.push_back(vector<int>(1,i));
            }
        }
        
        //open each document
        for (int j=0; j < int(train_filenames.size()); ++j) {
            ifstream infile(directory + train_filenames[j]);
            if(infile.is_open()) {
                //each line
                string line;
                while (getline(infile, line)) {
                    vector<string> parsed_line = word_tokenize(line);
                    //each word in document
                    for (string word: parsed_line) {
                        for (int i = 0; i < int(vocab.size()); ++i) {
                            if (word == vocab[i]) {
                                train_mat[j][i] = true;
                            }
                        }
                    }
                }
            }
            else {
                cout << "Error opening " << train_filenames[j] << endl;
            }
            infile.close();
        }
        
        //Test Matrix
        test_filenames = glob(directory + "*sample*", 1);
        test_mat.resize(int(test_filenames.size()), vector<bool>(int(vocab.size()),0));
        
        for (int i=0; i < int(test_filenames.size()); ++i) {
            
            //extract sample doc stopwords
            ifstream infile(directory + test_filenames[i]);
            if(infile.is_open()) {
                //each line
                string line;
                while (getline(infile, line)) {
                    vector<string> parsed_line = word_tokenize(line);
                    //each word in document
                    for (string word: parsed_line) {
                        for (int j = 0; j < int(vocab.size()); ++j) {
                            if (word == vocab[j]) {
                                test_mat[i][j] = true;
                            }
                        }
                    }
                }
            }
            else {
                cout << "Error opening " << test_filenames[i] << endl;
            }
            infile.close();
        }
        
    }

    // Function to print the confusion matrix.
    // Argument 1: "actual" is a list of integer class labels, one for each test example.
    // Argument 2: "predicted" is a list of integer class labels, one for each test example.
    // "actual" is the list of actual (ground truth) labels.
    // "predicted" is the list of labels predicted by your classifier.
    // "actual" and "predicted" MUST be in one-to-one correspondence.
    // That is, actual[i] and predicted[i] stand for testfile[i].
    void printConfMat(vector<int>&actual, vector<int>&predicted){
        vector<int> all_labels;
        assert(actual.size() == predicted.size());
        for (vector<int>::iterator i = actual.begin(); i != actual.end(); i++)
            all_labels.push_back((*i));
        for (vector<int>::iterator i = predicted.begin(); i != predicted.end(); i++)
            all_labels.push_back((*i));
        sort( all_labels.begin(), all_labels.end() );
        all_labels.erase( unique( all_labels.begin(), all_labels.end() ), all_labels.end() );
        map<pair<int,int>, unsigned> confmat;  // Confusion Matrix
        int itt = 0;
        for (vector<int>::iterator i = actual.begin(); i != actual.end(); i++){
            int a = (*i);
            pair<int, int> pp = make_pair(a, predicted[itt]);
            if (confmat.find(pp) == confmat.end()) confmat[pp] = 1;
            else confmat[pp] += 1;
            itt++;
        }
        cout << "\n\n";
        cout << "0 ";  // Actual labels column (aka first column)
        vector<int> tmp_labels;
        for (vector<int>::iterator i = all_labels.begin(); i != all_labels.end(); i++){
            int label2 = (*i);
            cout << label2 << " ";
            tmp_labels.push_back(label2);
        }
        cout << "\n";
        for (vector<int>::iterator i = all_labels.begin(); i != all_labels.end(); i++){
            int label = (*i);
            cout << label << " ";
            for (vector<int>::iterator i2 = tmp_labels.begin(); i2 != tmp_labels.end(); i2++){
                int label2 = (*i2);
                pair<int, int> pp = make_pair(label, label2);
                if (confmat.find(pp) == confmat.end()) cout << "0 ";
                else cout << confmat[pp] << " ";
            }
            cout << "\n";
        }
    }

    // Function to remove leading, trailing, and extra space from a string.
    // Inputs a string with extra spaces.
    // Outputs a string with no extra spaces.
    string remove_extra_space(string str){
        string buf; // Have a buffer string
        stringstream ss(str); // Insert the string into a stream
        vector<string> tokens; // Create vector to hold our words
        while (ss >> buf) tokens.push_back(buf);
        const char* const delim = " ";
        ostringstream imploded;
        copy(tokens.begin(), tokens.end(), ostream_iterator<string>(imploded, delim));
        return imploded.str();
    }

    // Tokenizer.
    // Input: string
    // Output: list of lowercased words from the string
    vector<string> word_tokenize(string input_string){
        string punctuations = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
        string extra_space_removed = remove_extra_space(input_string);
        string punctuation_removed = "";
        for (unsigned i = 0; i < extra_space_removed.length(); i++) {
            char curr = extra_space_removed[i];
            if (punctuations.find(curr) == string::npos) punctuation_removed += curr;
        }
        transform(punctuation_removed.begin(), punctuation_removed.end(), punctuation_removed.begin(), ::tolower);
        string buf; // Have a buffer string
        stringstream ss(punctuation_removed); // Insert the string into a stream
        vector<string> tokens; // Create vector to hold our words
        while (ss >> buf) tokens.push_back(buf);
        return tokens;
    }

};



class BernoulliNaiveBayes{

public:
    
    string train_test_dir;
    vector<string> vocab;
    PreProcess p;

    BernoulliNaiveBayes(string train_test_dirr, vector<string> &vocabb, PreProcess &pp)
        : train_test_dir(train_test_dirr), vocab(vocabb), p(pp) {}
    
    
    // Define Train function
    void train(map <int, double>& prior, map <string, map<int, double>>& condprob, unordered_set<int>& all_labels, vector<int> v_idx, int numtrain) {
        
        //for each label
        for (int label: all_labels) {
            
            prior[label] = double(p.Nc[label].size())/p.N;
            for (int t=0; t < numtrain; ++t) {
                double Nct = 0;
                
                //each document under label
                for (int mat_idx: p.Nc[label]) {
                    if (p.train_mat[mat_idx][v_idx[t]]) {
                        Nct++;
                    }
                }
                condprob[vocab[v_idx[t]]][label] = (Nct + 1)/(double(p.Nc[label].size()) + 2);
            }
        }
        
        /*
        TRAINBERNOULLINB(C, D)                              given a set of authors C and a set of training documents D
        1 V ← EXTRACTVOCABULARY(D)                          vocab
        2 N ← COUNTDOCS(D)                                  summ(all training docs)
        3 for each c ∈ C                                    for an author c     [for: all_labels]
        4     do Nc ← COUNTDOCSINCLASS(D, c)                        num training documents D with author c
        5         prior[c] ← Nc/N                                           prior[c]
        6         for each t ∈ V                                            for each stopword
        7             do Nct ← COUNTDOCSINCLASSCONTAININGTERM(D, c, t)              num training documents D with author c && stopword
        8                 condprob[t][c] ← (Nct + 1)/(Nc + 2)                       p(stopword | author) == for a stopword, author c -> return probability
        9 return V, prior, condprob
         
        */

    }

    // Define Test function
    int test(map <int, double>& prior, map <string, map<int, double>>& condprob, unordered_set<int>& all_labels, vector<int> v_idx, int numtrain, int file_idx){
        
        //find max probability of an author (label) given a document
        double maxp = -1*numeric_limits<double>::infinity();
        int idx = 0;
        
        //each author
        for (int c: all_labels) {
            double prob = log2(prior[c]);
            
            //add the log probabilities of each stopword
            for (int i=0; i < numtrain; ++i) {
                if (p.test_mat[file_idx][v_idx[i]]) { prob += log2(condprob[vocab[v_idx[i]]][c]); }
                else { prob += log2(1 - condprob[vocab[v_idx[i]]][c]); }
            }
            
            //check if prob > maxp
            if (prob > maxp) {
                maxp = prob;
                idx = c;
            }
        }
        
        /*
        
         APPLYBERNOULLINB(C, V, prior, condprob, d)                 return most likely author c
         1 Vd ← EXTRACTTERMSFROMDOC(V, d)                           SET of stopwords in sample (test) doc
         2 for each c ∈ C                                           for each author author c, maxp & idx
         3     do score[c] ← log prior[c]                               score_c = log2(prior[c])
         4     for each t ∈ V                                           for each label
         5         do if t ∈ Vd                                             if(stopword is in doc)
         6             then score[c] += log condprob[t][c]                      score += log(condprob[label][author c]
         7             else score[c] += log(1 − condprob[t][c])             update most likely author, maxp
         8 return arg maxc∈C                                        return idx
         score[c]
         
        */

        return idx;
    }
};

int main(int argc, char** argv) {
    
    //Check that file argument was used
    if (argc != 2) {
        cout << "Error check number of arguments" << endl;
        return 1;
    }
    
    //Preprocess stopwords
    //Input file
    freopen("stopwords.txt", "r", stdin);
    
    //go through all stopwords and insert into vocab vector
    vector<string> vocabb;
    string word;
    while (cin >> word) {
        vocabb.push_back(word);
    }
    fclose(stdin);
    
    //Create instance of Bernoulli Naive Bayes
    string dir = string(argv[1]);   //"/Users/MaxGittelman/Documents/eecs/492/Project3/Project3/" + string(argv[1]);
    if (dir[dir.size()-2] == 'G' || dir[dir.size()-2] == 'H') {
        g_h = true;
    }
    else {
        g_h = false;
    }
    unordered_set<int> all_labels;
    PreProcess pp(dir, vocabb, all_labels);
    BernoulliNaiveBayes bnb(dir, vocabb, pp);
    
    //Call train
    map <int, double> prior;
    map <string, map<int, double>> condprob;
    vector<int> v_idx(int(vocabb.size()));
    iota(v_idx.begin(), v_idx.end(), 0);
    bnb.train(prior, condprob, all_labels, v_idx, int(vocabb.size()));
    
    //Classify each sample file
    vector<int> predicted;
    vector<int> actual;
    string problem = string(argv[1]);
    char letter = problem[problem.size()-2];
    string line;
    ifstream infile("test_ground_truth.txt");
    if(infile.is_open()) {
        //read upto labels for letter
        while (getline(infile, line)) {
            if (line[7] == letter) {
                break;
            }
        }
        //read each line until a space
        actual.push_back((line[line.length()-2] - '0')*10 + (line[line.length()-1] - '0'));
        while (getline(infile, line)) {
            if (line == "") {
                break;
            }
            else {
                actual.push_back((line[line.length()-2] - '0')*10 + (line[line.length()-1] - '0'));
            }
        }
        infile.close();
    }
    else {
        cout << "Error opening ground truths" << endl;
    }
    vector<string> test_filenames = glob(dir + "*sample*", 1);
    
    for (int i=0; i < int(test_filenames.size()); ++i) {
        predicted.push_back(bnb.test(prior, condprob, all_labels, v_idx, int(vocabb.size()),i));
    }
    
    //print Accuracy
    double error = 0;
    if (predicted.size() != actual.size()) { assert(1); }
    for (int i = 0; i < int(predicted.size()); ++i) {
        if (predicted[i] != actual[i]) {
            error++;
        }
    }
    double accuracy = ((double(predicted.size())-error)/double(predicted.size()))*100;
    cout << endl;
    printf("%s%f%s", "Accuracy: ", accuracy, "%");
    
    //print Confusion Matrix
    pp.printConfMat(actual, predicted);
    cout << endl;
    
    //feature ranking
    class max_cmp
    {
    public:
        bool operator()(const pair<string, double> a, const pair<string, double> b) const
        {
            return a.second < b.second;
        }
    };
    priority_queue<pair<string, double>, vector<pair<string, double>>, max_cmp> entropy;
    for (string f: vocabb) {
        double sum = 0;
        for (int c: all_labels) {
            sum -= prior[c]*condprob[f][c]*log2(condprob[f][c]);
        }
        entropy.push(pair<string, double>(f,sum));
    }
    for (int i = 1; i < 21; ++i) {
        cout << i << " " << entropy.top().first << " " << entropy.top().second << endl;
        entropy.pop();
    }
    
    //feature frequency
    vector<pair<int, int>> frequency(int(vocabb.size()));
    vector<string> train_filenames = glob(dir + "*train*", 0);
    for (int i=0; i < int(vocabb.size()); ++i) {
        frequency[i].first = i;
        frequency[i].second = 0;
    }
    for (string file: train_filenames) {
        
        ifstream infile(dir + file);
        if(infile.is_open()) {
            string line;
            while (getline(infile, line)) {
                vector<string> parsed_line = pp.word_tokenize(line);
                for (string word: parsed_line) {
                    for (int i=0; i < int(vocabb.size()); ++i) {
                        if (vocabb[i] == word) {
                            frequency[i].second++;
                        }
                    }
                }
            }
        }
        else {
            cout << "Error opening " << file << endl;
        }
        infile.close();
    }
    
    //train top 10,20,...,420,423 features
    class freq_comp {
    public:
        bool operator() (const pair<int, int>& a, const pair<int, int>& b) const {
            return a.second > b.second;
        }
    };
    sort(frequency.begin(), frequency.end(), freq_comp());
    
    //train and test accuracy
    int last_x = int(vocabb.size());
    double last_y = accuracy;
    for (int i=0; i < int(vocabb.size()); ++i) {
        v_idx[i] = frequency[i].first;
    }
    vector<int> x;
    vector<double> y;
    cout << endl;
    for (int i=10; i < int(frequency.size()); i += 10) {
        
        BernoulliNaiveBayes bnb_f(dir, vocabb, pp);
        condprob.clear();
        bnb_f.train(prior, condprob, all_labels, v_idx, i);
        
        //test accuracy
        predicted.clear();
        for (int f=0; f < int(test_filenames.size()); ++f) {
            predicted.push_back(bnb_f.test(prior, condprob, all_labels, v_idx, i, f));
        }
        
        //print Accuracy
        double error = 0;
        if (predicted.size() != actual.size()) { assert(1); }
        for (int j = 0; j < int(predicted.size()); ++j) {
            if (predicted[j] != actual[j]) {
                error++;
            }
        }
        double accuracy = ((double(predicted.size())-error)/double(predicted.size()))*100;
        x.push_back(i);
        y.push_back(accuracy);
        cout << i << " Accuracy: " << accuracy << '%' << endl;
    }
    
    //calculate for final iterationf
    x.push_back(last_x);
    y.push_back(last_y);
    cout << last_x << " Accuracy: " << last_y << '%' << endl;
    
    return 0;
}
