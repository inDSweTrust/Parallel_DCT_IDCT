#include "cnpy/cnpy.h"
#include <iostream>
#include <omp.h>
#include <complex.h>
#include <cstdlib>
#include <stdlib.h>
#include <map>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>

using namespace std;

void print_npy_array(double* npydata, size_t nrows, size_t ncols, size_t nchannels) {

    for (size_t k = 0; k < nchannels; k++) {
        cout << "CHANNEL " << k + 1<< endl;
        for (size_t i = 0; i < nrows; i++) {
            for (size_t j = 0; j < ncols; j++) {
                cout << npydata[i*ncols*nchannels + j*nchannels + k] << " ";
            }
        cout << " " << endl;
        }
    }
}


void DCT(double* arr, double* dct, double* Q, size_t const nrows, size_t const ncols, size_t const nchannels) {

    
    double coef1 = 1 / sqrt(8);
    double coef2 = sqrt(2) / sqrt(8);

    for (size_t k = 0; k < nchannels; k++) {

        # pragma omp parallel for 
        for (size_t col=0; col < ncols; col += 8) {
            
            for (size_t row=0; row < nrows; row += 8) {

                for (size_t i=0; i < 8; i++) {

                    auto ii = row + i;

                    for (size_t j=0; j < 8; j++) {

                        auto jj = col + j;

                        double Li, Lj;
                        if (i == 0) {
                            Li = coef1;
                        }
                        else {
                            Li = coef2;
                        }
                        if (j == 0) {
                            Lj = coef1;
                        }
                        else {
                            Lj = coef2;
                        }
                        double sum = 0.0; double freq;
                        for (size_t x = 0; x < 8; x++) {

                            for (size_t y= 0; y < 8; y++) {

                                freq = arr[ii*ncols*nchannels + jj*nchannels + k] *
                                    cos((2 * x + 1) * i * M_PI / (2 * 8)) * 
                                    cos((2 * y + 1) * j * M_PI / (2 * 8));

                                sum += freq;
                            }
                        }

                        dct[ii*ncols*nchannels + jj*nchannels + k] = round( (Li * Lj * sum) / Q[i*8 + j]);
                    }
                }
            }
        }
    }
}



void IDCT(double* dct, double* idct, double* Q, size_t const nrows, size_t const ncols, size_t const nchannels) {

    double coef1 = 1 / sqrt(8);
    double coef2 = sqrt(2) / sqrt(8);

    for (size_t k = 0; k < nchannels; k++) {

        # pragma omp parallel for 
        for (size_t col=0; col < ncols; col += 8) {
            
            for (size_t row=0; row < nrows; row += 8) {

                for (size_t i=0; i < 8; i++) {

                    auto ii = row + i;

                    for (size_t j=0; j < 8; j++) {

                        auto jj = col + j;

                        double sum = 0.0; 
                        double freq; 
                        double Lx, Ly;
                        for (size_t x = 0; x < 8; x++) {
                            auto xx = row + x;
                            for (size_t y= 0; y < 8; y++) {
                                auto yy = col + y;

                                if (x == 0) {
                                    Lx = coef1;
                                }
                                else {
                                    Lx = coef2;
                                }
                                if (y == 0) {
                                    Ly = coef1;
                                }
                                else {
                                    Ly = coef2;
                                }

                                double dct_val = dct[xx*ncols*nchannels + yy*nchannels + k] * Q[x*8 + y];
                                freq = dct_val *
                                    cos((2 * i + 1) * x * M_PI / (2 * 8)) * 
                                    cos((2 * j + 1) * y * M_PI / (2 * 8));

                                sum += freq * Lx * Ly;
                            }
                        }
                        
                        idct[ii*ncols*nchannels + jj*nchannels + k] = sum;
                    }
                }
            }
        }
    }
}



int main (int argc, char *argv[]) {

    if(argc < 1){
        cerr << "Input npy file missing." << endl;
        exit(1);
    }

    string npyinput = argv[1];

    // Load Numpy array
    cnpy::NpyArray npyarr = cnpy::npy_load(npyinput);
    double* npydata = npyarr.data<double>();

    // Load Quantization table
    cnpy::NpyArray Qtable = cnpy::npy_load("Qt.npy");
    double* Q = Qtable.data<double>();


    size_t nrows = npyarr.shape[0];
    size_t ncols = npyarr.shape[1];
    size_t nchannels = npyarr.shape[2];
    int nthreads = 1;

    if(argc == 4 && strcasecmp(argv[2], "-t") == 0){  //@TODO: check if correct
        nthreads = atoi(argv[3]);
        omp_set_num_threads(nthreads);
    }

    cout << "nrows: " << nrows << endl;
    cout << "ncols: " << ncols << endl;
    cout << "channels: " << nchannels << endl;
    cout << "nthreads: " << nthreads << endl;

    // create arrays for intermediate results
    double* dct = new double[nrows * ncols * nchannels];
    double* idct = new double[nrows * ncols * nchannels];
    double* zigzag = new double[nrows * ncols * nchannels];

    // print_npy_array(npydata, nrows, ncols, nchannels);

    // Execute all functions 
    //*************************************
    auto t1 = omp_get_wtime();

    // Discrete Cosine Transform of image & Quantization
    DCT(npydata, dct, Q, nrows, ncols, nchannels);

    // Inverse DCT & Quantization
    IDCT(dct, idct, Q, nrows, ncols, nchannels);
    

    auto t2 = omp_get_wtime();
    //*************************************
    cout << "Execution time: " << (t2-t1) << endl;

    // Export to NumPy
    cnpy::npy_save("test_out_dct.npy", &dct[0], {nrows, ncols, nchannels},"w");
    cnpy::npy_save("test_out_idct.npy", &idct[0], {nrows, ncols, nchannels},"w");
    

    // free mem
    delete dct;
    delete idct;
    

}