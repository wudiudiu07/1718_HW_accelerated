#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace std;

unsigned char data_train[60000][784];
unsigned char data_test[10000][784];
unsigned char label_train[60000];
unsigned char label_test[10000];

float conv_w[5][49];
double conv_b[5][28][28];
double conv_layer[5][28][28];
double sig_layer[5][28][28];
char max_pooling[5][28][28];
double max_layer[5][14][14];
unsigned char img[1120];
int vector_y[10];

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

float dw_max[5][784];
double dw_conv[5][7][7];
double db_conv[5][28][28];

const double eta = 0.01;
/* ************************************************************ */
/* Helper functions */
int randint(int a) {
  return rand() % a;
}
void get_densemax(int a, double dmax[10]) {
  dmax = dense_softmax;
}
unsigned char get_img(int i) {
  return img[i];
}
float get_weight(int a, int i) {
  return conv_w[a][i];
}
void get_convb(double cb[5][28][28]) {
  cb = conv_b;
}

float convolution(int a, int b, unsigned char img[], int offset_img_a, int offset_img_b, float weight[49]) {
  float result = 0;
  for (int i = 0; i < a; i++) {
    for (int j = 0; j < b; j++) {
      result += img[(i + offset_img_a) * 32 + j + offset_img_b] * weight[i * b + j];
    }
  }
  return result;
}
double sigmoid(double x) {
  if (x > 500) x = 500;
  if (x < -500) x = -500;
  return 1 / (1 + exp(-x));
}
int set_sig_layer(float u, int f_dim, int i, int j) {
  sig_layer[f_dim][i][j] = sigmoid(u + conv_b[f_dim][i][j]);
  return 0;
}
double d_sigmoid(double x) {
  double sig = sigmoid(x);
  return sig * (1 - sig);
}
double softmax_den(double* x, int len) {
  double val = 0;
  for (int i = 0; i < len; i++) {
    val += exp(x[i]);
  }
  return val;
}

int initialise_weights() {
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 28; j++) {
      for (int k = 0; k < 28; k++) {
        if (j < 7 && k < 7) {
          conv_w[i][j * 7 + k] = 2 * double(rand()) / RAND_MAX - 1;
        }
        conv_b[i][j][k] = 2 * double(rand()) / RAND_MAX - 1;
      }
    }
  }

  for (int i = 0; i < 980; i++) {
    for (int j = 0; j < 120; j++) {
      dense_w[i][j] = 2 * double(rand()) / RAND_MAX - 1;
    }
  }
  for (int i = 0; i < 120; i++) {
    dense_b[i] = 2 * double(rand()) / RAND_MAX - 1;
  }

  for (int i = 0; i < 120; i++) {
    for (int j = 0; j < 10; j++) {
      dense_w2[i][j] = 2 * double(rand()) / RAND_MAX - 1;
    }
  }
  for (int i = 0; i < 10; i++) {
    dense_b2[i] = 2 * double(rand()) / RAND_MAX - 1;
  }
  return 0;
}
/* ************************************************************ */

/* ************************************************************ */

/* Forward Pass */
void forward_pass(unsigned char img[1120]) {
  const int filter_size = 7;

  // Convolution Operation + Sigmoid Activation
  for (int filter_dim = 0; filter_dim < 5; filter_dim++) {
    for (int i = 0; i < 28; i++) {
      for (int j = 0; j < 28; j++) {
        max_pooling[filter_dim][i][j] = 0;

        conv_layer[filter_dim][i][j] = 0;
        sig_layer[filter_dim][i][j] = 0;
        conv_layer[filter_dim][i][j] = convolution(filter_size, filter_size, img, i + 1, j - 2, conv_w[filter_dim]);
        sig_layer[filter_dim][i][j] = sigmoid(conv_layer[filter_dim][i][j] + conv_b[filter_dim][i][j]);
      }
    }
  }

  // MAX Pooling (max_pooling, max_layer)
  double cur_max = 0;
  int max_i = 0, max_j = 0;
  for (int filter_dim = 0; filter_dim < 5; filter_dim++) {
    for (int i = 0; i < 28; i += 2) {
      for (int j = 0; j < 28; j += 2) {
        max_i = i;
        max_j = j;
        cur_max = sig_layer[filter_dim][i][j];
        for (int k = 0; k < 2; k++) {
          for (int l = 0; l < 2; l++) {
            if (sig_layer[filter_dim][i + k][j + l] > cur_max) {
              max_i = i + k;
              max_j = j + l;
              cur_max = sig_layer[filter_dim][max_i][max_j];
            }
          }
        }
        max_pooling[filter_dim][max_i][max_j] = 1;
        max_layer[filter_dim][i / 2][j / 2] = cur_max;
      }
    }
  }

  int k = 0;
  for (int filter_dim = 0; filter_dim < 5; filter_dim++) {
    for (int i = 0; i < 14; i++) {
      for (int j = 0; j < 14; j++) {
        dense_input[k] = max_layer[filter_dim][i][j];
        k++;
      }
    }
  }

  // Dense Layer
  for (int i = 0; i < 120; i++) {
    dense_sum[i] = 0;
    dense_sigmoid[i] = 0;
    for (int j = 0; j < 980; j++) {
      dense_sum[i] += dense_w[j][i] * dense_input[j];
    }
    dense_sum[i] += dense_b[i];
    dense_sigmoid[i] = sigmoid(dense_sum[i]);
  }

  // Dense Layer 2
  for (int i = 0; i < 10; i++) {
    dense_sum2[i] = 0;
    for (int j = 0; j < 120; j++) {
      dense_sum2[i] += dense_w2[j][i] * dense_sigmoid[j];
    }
    dense_sum2[i] += dense_b2[i];
  }

  // Softmax Output
  double den = softmax_den(dense_sum2, 10);
  for (int i = 0; i < 10; i++) {
    dense_softmax[i] = exp(dense_sum2[i]) / den;
  }
}

int update_weights(int a) {
  for (int i = 0; i < 120; i++) {
    dense_b[i] -= eta * db1[i];
    for (int j = 0; j < 10; j++) {
      dense_b2[j] -= eta * db2[j];
      dense_w2[i][j] -= eta * dw2[i][j];
    }
    for (int k = 0; k < 980; k++) {
      dense_w[k][i] -= eta * dw1[k][i];
    }
  }

  for (int i = 0; i < 5; i++) {
    for (int k = 0; k < 7; k++) {
      for (int j = 0; j < 7; j++) {
        conv_w[i][k * 7 + j] -= eta * dw_conv[i][k][j];
      }
    }
    for (int l = 0; l < 28; l++) {
      for (int m = 0; m < 28; m++) {
        conv_b[i][l][m] -= eta * db_conv[i][l][m];
      }
    }
  }
  return 0;
}

int forward_pass_noconv(int a) {
  // MAX Pooling (max_pooling, max_layer)
  for (int filter_dim = 0; filter_dim < 5; filter_dim++) {
    for (int i = 0; i < 28; i++) {
      for (int j = 0; j < 28; j++) {
        max_pooling[filter_dim][i][j] = 0;
      }
    }
  }
  double cur_max = 0;
  int max_i = 0, max_j = 0;
  for (int filter_dim = 0; filter_dim < 5; filter_dim++) {
    for (int i = 0; i < 28; i += 2) {
      for (int j = 0; j < 28; j += 2) {
        max_i = i;
        max_j = j;
        cur_max = sig_layer[filter_dim][i][j];
        for (int k = 0; k < 2; k++) {
          for (int l = 0; l < 2; l++) {
            if (sig_layer[filter_dim][i + k][j + l] > cur_max) {
              max_i = i + k;
              max_j = j + l;
              cur_max = sig_layer[filter_dim][max_i][max_j];
            }
          }
        }
        max_pooling[filter_dim][max_i][max_j] = 1;
        max_layer[filter_dim][i / 2][j / 2] = cur_max;
      }
    }
  }

  int k = 0;
  for (int filter_dim = 0; filter_dim < 5; filter_dim++) {
    for (int i = 0; i < 14; i++) {
      for (int j = 0; j < 14; j++) {
        dense_input[k] = max_layer[filter_dim][i][j];
        k++;
      }
    }
  }

  // Dense Layer
  for (int i = 0; i < 120; i++) {
    dense_sum[i] = 0;
    dense_sigmoid[i] = 0;
    for (int j = 0; j < 980; j++) {
      dense_sum[i] += dense_w[j][i] * dense_input[j];
    }
    dense_sum[i] += dense_b[i];
    dense_sigmoid[i] = sigmoid(dense_sum[i]);
  }

  // Dense Layer 2
  for (int i = 0; i < 10; i++) {
    dense_sum2[i] = 0;
    for (int j = 0; j < 120; j++) {
      dense_sum2[i] += dense_w2[j][i] * dense_sigmoid[j];
    }
    dense_sum2[i] += dense_b2[i];
  }

  // Softmax Output
  double den = softmax_den(dense_sum2, 10);
  for (int i = 0; i < 10; i++) {
    dense_softmax[i] = exp(dense_sum2[i]) / den;
  }
  return 0;
}
/* ************************************************************ */

/* ************************************************************ */
/* Backward Pass */
int backward_pass(int a) {
  double delta4[10];
  for (int i = 0; i < 10; i++) {
    delta4[i] = dense_softmax[i] - vector_y[i];  // Derivative of Softmax + Cross entropy
    db2[i] = delta4[i];                          // Bias Changes
  }

  // Calculate Weight Changes for Dense Layer 2
  for (int i = 0; i < 120; i++) {
    for (int j = 0; j < 10; j++) {
      dw2[i][j] = dense_sigmoid[i] * delta4[j];
    }
  }

  // Delta 3
  double delta3[120];
  for (int i = 0; i < 120; i++) {
    delta3[i] = 0;
    for (int j = 0; j < 10; j++) {
      delta3[i] += dense_w2[i][j] * delta4[j];
    }
    delta3[i] *= d_sigmoid(dense_sum[i]);
  }
  for (int i = 0; i < 120; i++) db1[i] = delta3[i];  // Bias Weight change

  // Calculate Weight Changes for Dense Layer 1
  for (int i = 0; i < 980; i++) {
    for (int j = 0; j < 120; j++) {
      dw1[i][j] = dense_input[i] * delta3[j];
    }
  }

  // Delta2
  double delta2[980];
  for (int i = 0; i < 980; i++) {
    delta2[i] = 0;
    for (int j = 0; j < 120; j++) {
      delta2[i] += dense_w[i][j] * delta3[j];
    }
    delta2[i] *= d_sigmoid(dense_input[i]);
  }

  // Calc back-propagated max layer dw_max
  int k = 0;
  for (int filter_dim = 0; filter_dim < 5; filter_dim++) {
    for (int i = 0; i < 28; i += 2) {
      for (int j = 0; j < 28; j += 2) {
        for (int l = 0; l < 2; l++) {
          for (int m = 0; m < 2; m++) {
            if (max_pooling[filter_dim][i + l][j + m] == 1)
              dw_max[filter_dim][i * 28 + j] = delta2[k];
            else
              dw_max[filter_dim][i * 28 + j] = 0;
          }
        }
        k++;
      }
    }
  }
  // Calc Conv Bias Changes
  for (int filter_dim = 0; filter_dim < 5; filter_dim++) {
    for (int i = 0; i < 28; i++) {
      for (int j = 0; j < 28; j++) {
        db_conv[filter_dim][i][j] = dw_max[filter_dim][i * 28 + j];
      }
    }
  }

  // Set Conv Layer Weight changes to 0
  for (int filter_dim = 0; filter_dim < 5; filter_dim++) {
    for (int i = 0; i < 7; i++) {
      for (int j = 0; j < 7; j++) {
        dw_conv[filter_dim][i][j] = 0;
      }
    }
  }

  // Calculate Weight Changes for Conv Layer
  for (int filter_dim = 0; filter_dim < 5; filter_dim++) {
    for (int k = 0; k < 7; k++) {
      for (int l = 0; l < 7; l++) {
        dw_conv[filter_dim][k][l] = convolution(28, 28, img, k + 1, l - 2, dw_max[filter_dim]);
      }
    }
  }
  return 0;
}
/* ************************************************************ */

int read_train_data() {
  ifstream csvread;
  csvread.open("./mnist_train.csv", ios::in);
  if (csvread) {
    string s;
    int data_pt = 0;
    while (getline(csvread, s)) {
      stringstream ss(s);
      int pxl = 0;
      while (ss.good()) {
        string substr;
        getline(ss, substr, ',');
        if (pxl == 0) {
          label_train[data_pt] = stoi(substr);
        } else {
          data_train[data_pt][pxl - 1] = stoi(substr);
        }
        pxl++;
      }
      data_pt++;
    }
    csvread.close();
  } else {
    cerr << "Unable to read train data!" << endl;
    exit(EXIT_FAILURE);
  }
  return 0;
}
int read_test_data() {
  ifstream csvread;
  csvread.open("./mnist_test.csv", ios::in);
  if (csvread) {
    string s;
    int data_pt = 0;
    while (getline(csvread, s)) {
      stringstream ss(s);
      int pxl = 0;
      while (ss.good()) {
        string substr;
        getline(ss, substr, ',');
        if (pxl == 0) {
          label_test[data_pt] = stoi(substr);
        } else {
          data_test[data_pt][pxl - 1] = stoi(substr);
        }
        pxl++;
      }
      data_pt++;
    }
    csvread.close();
  } else {
    cerr << "Unable to read test data!" << endl;
    exit(EXIT_FAILURE);
  }
  return 0;
}
void give_img(unsigned char* vec) {
  int k = 0;
  for (int i = 0; i < 35; i++) {
    for (int j = 0; j < 32; j++) {
      if (i < 5 || j < 2 || i > 32 || j > 29) {
        img[i * 32 + j] = 0;
      } else {
        img[i * 32 + j] = vec[k++];
      }
    }
  }
}

int give_img_train(int randnum) {
  unsigned char* vec = 0;
  vec = data_train[randnum];
  int k = 0;
  for (int i = 0; i < 35; i++) {
    for (int j = 0; j < 32; j++) {
      if (i < 5 || j < 2 || i > 32 || j > 29) {
        img[i * 32 + j] = 0;
      } else {
        img[i * 32 + j] = vec[k++];
      }
    }
  }
  return 0;
}

int give_y(int randnum) {
  int y = label_train[randnum];
  for (int i = 0; i < 10; i++) vector_y[i] = 0;
  vector_y[y] = 1;
  return 0;
}
int give_prediction() {
  double max_val = dense_softmax[0];
  int max_pos = 0;
  for (int i = 1; i < 10; i++) {
    if (dense_softmax[i] > max_val) {
      max_val = dense_softmax[i];
      max_pos = i;
    }
  }

  return max_pos;
}

int filter_ite(unsigned char img1[1120], float weight[49], int filter_dim) {
  const int filter_size = 7;
  double conv_layer1 = 0;
  for (int i = 0; i < 28; i++) {
    for (int j = 0; j < 28; j++) {
      conv_layer1 = convolution(filter_size, filter_size, img1, i + 1, j - 2, weight);
      set_sig_layer(conv_layer1, filter_dim, i, j);
      // sig_layer[filter_dim][i][j] = sigmoid(conv_layer[filter_dim][i][j] + conv_b[filter_dim][i][j]);
    }
  }
  return 0;
}
int conv_sig(int a) {
  const int filter_size = 7;
  double conv_layer1 = 0;
  // unsigned char *img1 = 0;
  // double * weight = 0;
  // get_img(img1);
  //  Convolution Operation + Sigmoid Activation
  for (int filter_dim = 0; filter_dim < 5; filter_dim++) {
    for (int i = 0; i < 28; i++) {
      for (int j = 0; j < 28; j++) {
        max_pooling[filter_dim][i][j] = 0;

        conv_layer[filter_dim][i][j] = 0;
        sig_layer[filter_dim][i][j] = 0;
        conv_layer1 = convolution(filter_size, filter_size, img, i + 1, j - 2, conv_w[filter_dim]);
        set_sig_layer(conv_layer1, filter_dim, i, j);
        // sig_layer[filter_dim][i][j] = sigmoid(conv_layer[filter_dim][i][j] + conv_b[filter_dim][i][j]);
      }
    }
  }
}

void train(int i) {
  int num = 0;
  const int batch_size = 200;
  cout << "Epoch " << i << " done." << endl;
  for (int j = 0; j < batch_size; j++) {
    num = rand() % 60000;
    // int vector_y[10];
    give_y(num);
    give_img_train(num);

    // give_img(data_train[num]);
    // forward_pass(img);

    conv_sig(1);
    forward_pass_noconv(1);
    backward_pass(1);
    update_weights(1);
  }
}

int validate(int i, int cor) {
  cout << "validate " << i << " done." << endl;
  give_img(data_test[i]);
  forward_pass(img);
  int pre = give_prediction();
  if (pre == label_test[i]) cor++;
  return cor;
}

int main() {
  read_test_data();
  read_train_data();
  initialise_weights();

  int epoch = 500;

  cout << "Start Training." << endl;
  for (int i = 0; i < epoch; i++) {
    train(i);
  }

  int val_len = 600;
  int cor = 0;
  //   int confusion_mat[10][10];
  //   for (int i = 0; i < 10; i++) {
  //     for (int j = 0; j < 10; j++) confusion_mat[i][j] = 0;
  //   }

  cout << "Start Testing." << endl;
  for (int i = 0; i < val_len; i++) {
    cor = validate(i, cor);
    //     confusion_mat[label_test[i]][pre]++;
  }
  float accu = double(cor) / val_len;
  cout << "Accuracy: " << accu << endl;

  //   cout << "   0 1 2 3 4 5 6 7 8 9" << endl;
  //   for (int i = 0; i < 10; i++) {
  //     cout << i << ": ";
  //     for (int j = 0; j < 10; j++) {
  //       cout << confusion_mat[i][j] << " ";
  //     }
  //     cout << endl;
  //   }

  return 0;
}