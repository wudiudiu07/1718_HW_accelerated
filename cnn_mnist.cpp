#include <iostream>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <fstream>

#include <numeric>
#include <immintrin.h>

using namespace std;
const int filter_size=7;
const double eta=0.01;
const double eta_a[4] = {eta,eta,eta,eta};
const int batch_size=200;

unsigned char data_train[60000][784];
unsigned char data_test[10000][784];
unsigned char label_train[60000];
unsigned char label_test[10000];

double conv_w[5][7][8];
double conv_b[5][28][28];
double conv_layer[5][28][28];
double sig_layer[5][28][28];
char max_pooling[5][28][28];
double max_layer[5][14][14];

double dense_input[980];
double dense_w[980][120];
double dense_b[120];
double dense_sum[120];
double dense_sigmoid[120];
double dense_w2[120][10];
double dense_b2[10];
double dense_sum2[10];
double dense_softmax[10];

double dw2[120][10];
double db2[10];
double dw1[980][120];
double db1[120];

double dw_max[5][28][28];
double dw_conv[5][7][7];
double db_conv[5][28][28];

double conv_layer_tmp_array[4];


/* ************************************************************ */
/* Helper functions */
double sigmoid(double x) {
        if (x>500) x=500;
        if (x<-500) x=-500;
        return 1/(1+exp(-x));
}
double d_sigmoid(double x) {
        double sig = sigmoid(x);
        return sig*(1-sig);
}
double softmax_den(double *x, int len) {
        double val =0;
        for (int i=0; i<len; i++) {
                val += exp(x[i]);
        }
        return val;
}

void initialise_weights() {
        for (int i=0; i<5; i++) {
                for (int j=0; j<28; j++) {
                        for (int k=0; k<28; k++) {
                                if (j<7 && k<7) {
                                        conv_w[i][j][k] = 2*double(rand())/RAND_MAX-1;
                                }
                                conv_b[i][j][k] = 2*double(rand())/RAND_MAX-1;
                        }
                }
        }

        for (int i=0; i<980; i++) {
                for (int j=0; j<120; j++) {
                        dense_w[i][j] = 2*double(rand()) / RAND_MAX-1;
                }
        }
        for (int i=0; i<120; i++) {
                dense_b[i] = 2*double(rand()) / RAND_MAX-1;
        }

        for (int i=0; i<120; i++) {
                for (int j=0; j<10; j++) {
                        dense_w2[i][j] = 2*double(rand())/RAND_MAX-1;
                }
        }
        for (int i=0; i<10; i++) {
                dense_b2[i] = 2*double(rand())/RAND_MAX-1;
        }
}
/* ************************************************************ */

/* ************************************************************ */
/* Forward Pass */
void forward_pass(double img[][33]) {
	    __m256d img_tmp_a, conv_w_tmp_a, img_tmp_b, conv_w_tmp_b, product_tmp_a, product_tmp_b;
        // Convolution Operation + Sigmoid Activation
        for (int filter_dim=0; filter_dim<5; filter_dim++) {
                for (int i=0; i<28; i++) {
                        for (int j=0; j<28; j++) {
                                max_pooling[filter_dim][i][j] = 0;

                                conv_layer[filter_dim][i][j] = 0;
                                sig_layer[filter_dim][i][j] = 0;
                                // for (int k=0; k<filter_size; k++) {
                                //         for (int l=0; l<filter_size; l++) {
                                //                 conv_layer[filter_dim][i][j] += img[i+k+1][j+l-2]*conv_w[filter_dim][k][l];
                                //         }
                                // }
                                // sig_layer[filter_dim][i][j] = sigmoid(conv_layer[filter_dim][i][j] + conv_b[filter_dim][i][j]);
								for (int k=0; k<filter_size; k++) {
									    int A = i+k+1;
									    img[A][32] = 0;
									    conv_w[filter_dim][k][7] = 0;
                                        // Partial unroll makes this faster
										img_tmp_a = _mm256_set_pd(img[A][j+1],img[A][j],img[A][j-1],img[A][j-2]);
										conv_w_tmp_a = _mm256_set_pd(conv_w[filter_dim][k][3],conv_w[filter_dim][k][2],conv_w[filter_dim][k][1],conv_w[filter_dim][k][0]);
										img_tmp_b = _mm256_set_pd(img[A][j+5],img[A][j+4],img[A][j+3],img[A][j+2]);
										conv_w_tmp_b = _mm256_set_pd(conv_w[filter_dim][k][7],conv_w[filter_dim][k][6],conv_w[filter_dim][k][5],conv_w[filter_dim][k][4]);
										product_tmp_a = _mm256_mul_pd(img_tmp_a, conv_w_tmp_a);
										_mm256_storeu_pd(conv_layer_tmp_array,product_tmp_a);
										conv_layer[filter_dim][i][j] += conv_layer_tmp_array[3] + conv_layer_tmp_array[2] + conv_layer_tmp_array[1] + conv_layer_tmp_array[0];
										product_tmp_b = _mm256_mul_pd(img_tmp_b, conv_w_tmp_b);
										_mm256_storeu_pd(conv_layer_tmp_array,product_tmp_b);
										conv_layer[filter_dim][i][j] += conv_layer_tmp_array[3] + conv_layer_tmp_array[2] + conv_layer_tmp_array[1] + conv_layer_tmp_array[0];
                                }
								// _mm256_storeu_pd(&conv_layer[filter_dim][i][j],conv_layer_tmp);
                                sig_layer[filter_dim][i][j] = sigmoid(conv_layer[filter_dim][i][j] + conv_b[filter_dim][i][j]);
                        }
                }
        }

        // MAX Pooling (max_pooling, max_layer)
        double cur_max =0;
        int max_i=0, max_j=0;
        for (int filter_dim=0; filter_dim<5; filter_dim++) {
                for (int i=0; i<28; i+=2) {
                        for (int j=0; j<28; j+=2) {
                                max_i=i;
                                max_j=j;
                                cur_max=sig_layer[filter_dim][i][j];
                                for (int k=0; k<2; k++) {
                                        for (int l=0; l<2; l++) {
                                                if (sig_layer[filter_dim][i+k][j+l] > cur_max) {
                                                        max_i = i+k;
                                                        max_j = j+l;
                                                        cur_max = sig_layer[filter_dim][max_i][max_j];
                                                }
                                        }
                                }
                                max_pooling[filter_dim][max_i][max_j] = 1;
                                max_layer[filter_dim][i/2][j/2] = cur_max;
                        }
                }
        }
        int k=0;
        for (int filter_dim=0;filter_dim<5;filter_dim++) {
                for (int i=0;i<14;i++) {
                        for (int j=0;j<14;j++) {
                                dense_input[k] = max_layer[filter_dim][i][j];
                                k++;
                        }
                }
        }

        // Dense Layer
	//double dense_tmp[980];
	__m256d densew_tmp, tmp, sum_tmp;
	double dense_sum_a[4];
        for (int i=0; i<120; i++) {
                dense_sum[i] = 0;
                dense_sigmoid[i] = 0;
		sum_tmp = _mm256_setzero_pd();
		for (int j=0; j<980; j+=4) {
			//dense_sum[i] += dense_w[j][i] * dense_input[j];
			densew_tmp = _mm256_set_pd(dense_w[j+3][i], dense_w[j+2][i], dense_w[j+1][i],dense_w[j][i]);
			tmp = _mm256_mul_pd(densew_tmp, _mm256_loadu_pd(&dense_input[j]));
			sum_tmp = _mm256_add_pd(sum_tmp, tmp);
                }

		_mm256_storeu_pd(dense_sum_a, sum_tmp);
		dense_sum[i] = dense_b[i]+ dense_sum_a[3]+ dense_sum_a[2]+ dense_sum_a[1] + dense_sum_a[0];
		//dense_sum[i] += dense_b[i];
                dense_sigmoid[i] = sigmoid(dense_sum[i]);
        }

        // Dense Layer 2
        for (int i=0; i<10; i++) {
                dense_sum2[i]=0;
                for (int j=0; j<120; j++) {
                        dense_sum2[i] += dense_w2[j][i] * dense_sigmoid[j];
                }
                dense_sum2[i] += dense_b2[i];
        }

        // Softmax Output
        double den = softmax_den(dense_sum2, 10);
        for (int i=0; i<10; i++) {
                dense_softmax[i] = exp(dense_sum2[i])/den;
        }
}

void update_weights() {
	__m256d aeta = _mm256_set_pd(eta,eta,eta,eta);
	__m256d tmp;
//        for (int i=0; i<980; i++) {
//               for (int k=0; k<120; k+=4) {
//			//dense_w[i][k] -= eta*dw1[i][k];
//			tmp = _mm256_mul_pd(aeta, _mm256_loadu_pd(&dw1[i][k]));
//			_mm256_storeu_pd(&dense_w[i][k], _mm256_addsub_pd(_mm256_loadu_pd(&dense_w[i][k]), tmp));
//                }
//		if (i > 118)continue;
//
//                dense_b[i] -= eta*db1[i];
//
//                for (int j=0; j<10; j++) {
//			dense_b2[j] char-= eta*db2[j];
//                        dense_w2[i][j] -= eta*dw2[i][j];
//
//			//tmp = _mm256_mul_pd(aeta, _mm256_loadu_pd(&db2[j*4]));
//			//_mm256_storeu_pd(&dense_b2[j*4], _mm256_addsub_pd(_mm256_loadu_pd(&dense_b2[j*4]), tmp));
//
//
//			//tmp = _mm256_mul_pd(aeta, _mm256_loadu_pd(&dw2[i][j*4]));
//			//_mm256_storeu_pd(&dense_w2[i][j*4], _mm256_addsub_pd(_mm256_loadu_pd(&dense_w2[i][j*4]), tmp));
//
//			//dense_b2[8+j] -= eta*db2[8+j];
//                        //dense_w2[i][8+j] -= eta*dw2[i][8+j];
//                }
//
//        }
//
	for (int i=0; i<120; i++) {
                dense_b[i] -= eta*db1[i];
                for (int j=0; j<10; j++) {
                        dense_b2[j] -= eta*db2[j];
                        dense_w2[i][j] -= eta*dw2[i][j];
                }
//		for (int k=0; k<980; k++){
//			dense_w[k][i] -= eta*dw1[k][i];
//		}
	}
	for (int i=0; i<980; i++){
                for (int k=0; k<120; k+=4) {
			tmp = _mm256_mul_pd(aeta, _mm256_loadu_pd(&dw1[i][k]));
			_mm256_storeu_pd(&dense_w[i][k], _mm256_sub_pd(_mm256_loadu_pd(&dense_w[i][k]), tmp));
                        //dense_w[i][k] -= eta*dw1[i][k];

                }
        }

        for (int i=0; i<5; i++) {
                for (int k=0; k<7; k++) {
                        for (int j=0; j<7; j++) {
                                conv_w[i][k][j] -= eta*dw_conv[i][k][j];
                        }
                }
                for (int l=0; l<28; l++) {
                        for (int m=0; m<28; m+=4) {
				//conv_b[i][l][m] -= eta*db_conv[i][l][m];
				tmp = _mm256_mul_pd(aeta, _mm256_loadu_pd(&db_conv[i][l][m]));
				_mm256_storeu_pd(&conv_b[i][l][m], _mm256_sub_pd(_mm256_loadu_pd(&conv_b[i][l][m]), tmp));
                        }
                }
        }
}
/* ************************************************************ */

/* ************************************************************ */
/* Backward Pass */
void backward_pass(double *y_hat, int *y, double img[][33]) {
        double delta4[10];
        __m256d delta3_tmp,tmp,dense_temp,densew_tmp,sum_tmp;
        double delta2_tmp[4];
        double d_dense;
        for (int i=0; i<10; i++) {
                delta4[i] = y_hat[i] - y[i]; // Derivative of Softmax + Cross entropy
                db2[i] = delta4[i]; // Bias Changes
        }

        // Calculate Weight Changes for Dense Layer 2
        for (int i=0; i<120; i++) {
                for (int j=0; j<10; j++) {
                        dw2[i][j] = dense_sigmoid[i]*delta4[j];
                }
        }

        // Delta 3
        double delta3[120];
        for (int i=0; i<120; i++) {
                delta3[i] = 0;
                for (int j=0; j<10; j++) {
                        delta3[i] += dense_w2[i][j]*delta4[j];
                }
                delta3[i] *= d_sigmoid(dense_sum[i]);
        }
        for (int i=0; i<120; i++) db1[i] = delta3[i]; // Bias Weight change

        // Calculate Weight Changes for Dense Layer 1
        for (int i=0; i<980; i++) {
                for (int j=0; j<120; j+=4) {
                        delta3_tmp = _mm256_set_pd(delta3[j+3],delta3[j+2],delta3[j+1],delta3[j]);
                        dense_temp = _mm256_set_pd(dense_input[i],dense_input[i],dense_input[i],dense_input[i]);
                        tmp = _mm256_mul_pd(dense_temp, delta3_tmp);
                        //dw1[i][j] = dense_input[i]*delta3[j];
                        _mm256_storeu_pd(&dw1[i][j],tmp);
                }
        }

        // Delta2
        double delta2[980];
        for (int i=0; i<980; i++) {
                delta2[i] = 0;
				sum_tmp = _mm256_set_pd(0, 0, 0, 0);
                for (int j=0; j<120; j+=4) {
                        densew_tmp = _mm256_set_pd(dense_w[i][j+3],dense_w[i][j+2],dense_w[i][j+1],dense_w[i][j]);
                        tmp = _mm256_mul_pd(densew_tmp,_mm256_loadu_pd(&delta3[j]));
                        sum_tmp = _mm256_add_pd(sum_tmp,tmp);
                        //delta2[i] += dense_w[i][j]*delta3[j];
                }
                _mm256_storeu_pd(delta2_tmp,sum_tmp);
                d_dense = d_sigmoid(dense_input[i]);
                delta2[i] = (delta2_tmp[3]+delta2_tmp[2]+delta2_tmp[1]+delta2_tmp[0]) * d_dense;
                //delta2[i] *= d_sigmoid(dense_input[i]);
        }

        // Calc back-propagated max layer dw_max
        int k=0;
        for (int filter_dim=0; filter_dim<5; filter_dim++) {
                for (int i=0; i<28; i+= 2) {
                        for (int j=0; j<28; j+= 2) {
                                for (int l=0; l<2; l++) {
                                        for (int m=0; m<2; m++) {
                                                if (max_pooling[filter_dim][i+l][j+m] == 1) dw_max[filter_dim][i][j] = delta2[k];
                                                else dw_max[filter_dim][i][j] = 0;
                                        }
                                }
                                k++;
                        }
                }
        }
        // Calc Conv Bias Changes
        for (int filter_dim=0; filter_dim<5; filter_dim++) {
                for (int i=0; i<28; i++) {
                        for (int j=0; j<28; j++) {
                                db_conv[filter_dim][i][j] = dw_max[filter_dim][i][j];
                        }
                }
        }

        // Set Conv Layer Weight changes to 0
        for (int filter_dim=0; filter_dim<5; filter_dim++) {
                for (int i=0; i<7; i++) {
                        for (int j=0; j<7; j++) {
                                dw_conv[filter_dim][i][j] = 0;
                        }
                }
        }

        // Calculate Weight Changes for Conv Layer
        for (int filter_dim=0; filter_dim<5; filter_dim++) {
                for (int i=0; i<28; i++) {
                        for (int j=0; j<28; j++) {
                                double cur_val = dw_max[filter_dim][i][j];
                                for (int k=0; k<7; k++) {
                                        for (int l=0; l<7; l++) {
                                                dw_conv[filter_dim][k][l] += img[i+k+1][j+l-2] * cur_val;
                                        }
                                }
                        }
                }
        }


}
/* ************************************************************ */


void read_train_data() {
        ifstream csvread;
        csvread.open("/cad2/ece1718s/mnist_train.csv", ios::in);
        if(csvread) {
                string s;
                int data_pt = 0;
                while(getline(csvread, s)) {
                        stringstream ss(s);
                        int pxl = 0;
                        while( ss.good() ) {
                                string substr;
                                getline(ss, substr,',');
                                if (pxl == 0) {
                                        label_train[data_pt] = stoi(substr);
                                } else {
                                        data_train[data_pt][pxl-1] = stoi(substr);
                                }
                                pxl++;
                        }
                        data_pt++;
                }
                csvread.close();
        }
        else{
                cerr << "Unable to read train data!" << endl;
        exit (EXIT_FAILURE);
        }
}
void read_test_data() {
        ifstream csvread;
        csvread.open("/cad2/ece1718s/mnist_test.csv", ios::in);
        if(csvread) {
                string s;
                int data_pt = 0;
                while(getline(csvread, s)) {
                        stringstream ss(s);
                        int pxl = 0;
                        while( ss.good() ) {
                                string substr;
                                getline(ss, substr,',');
                                if (pxl == 0) {
                                        label_test[data_pt] = stoi(substr);
                                } else {
                                        data_test[data_pt][pxl-1] = stoi(substr);
                                }
                                pxl++;
                        }
                        data_pt++;
                }
                csvread.close();
        }
        else{
                cerr << "Unable to read test data!" << endl;
        exit (EXIT_FAILURE);
        }
}

void give_img(unsigned char* vec ,double img[][33]) {
        int k=0;
        for (int i=0; i<35; i++) {
                for (int j=0; j<32; j++) {
                        if (i<5 || j<2 || i>32 || j>29) {
                                img[i][j] = 0;
                        } else {
                                img[i][j] = vec[k++];
                        }
                }
        }
}

void give_y(int y, int *vector_y) {
        for (int i=0; i<10; i++) vector_y[i] =0;
        vector_y[y]=1;
}
int give_prediction() {
        double max_val = dense_softmax[0];
        int max_pos = 0;
        for (int i=1; i<10; i++) {
                if (dense_softmax[i] > max_val) {
                        max_val = dense_softmax[i];
                        max_pos = i;
                }
        }

        return max_pos;
}

int main() {
        read_test_data();
        read_train_data();
        initialise_weights();

        int epoch = 500;
        int num = 0;
        cout << "Start Training." << endl;
        for (int i=0; i<epoch; i++) {
                cout << "Epoch " << i << " done." << endl;
                for (int j=0; j<batch_size; j++) {
                        num = rand()%60000;
                        double img[35][33];
                        int vector_y[10];
                        give_y(label_train[num], vector_y);
                        give_img(data_train[num], img);
                        forward_pass(img);
                        backward_pass(dense_softmax, vector_y, img);
                        update_weights();
                }
        }

        int val_len = 600;
        int cor=0;
        int confusion_mat[10][10];
        for (int i=0; i<10; i++){
                for (int j=0; j<10; j++) confusion_mat[i][j] = 0;
        }

        cout << "Start Testing." << endl;
        for (int i=0; i<val_len; i++) {
                double img[35][33];
                give_img(data_test[i], img);
                forward_pass(img);
                int pre = give_prediction();
                confusion_mat[label_test[i]][pre]++;
                if (pre == label_test[i]) cor++;
        }
        float accu = double(cor)/val_len;
        cout << "Accuracy: " << accu << endl;

        cout << "   0 1 2 3 4 5 6 7 8 9" << endl;
        for (int i=0; i<10; i++){
                cout << i << ": ";
                for (int j=0; j<10; j++) {
                        cout << confusion_mat[i][j] << " ";
                }
                cout << endl;
        }

        return 0;
}
