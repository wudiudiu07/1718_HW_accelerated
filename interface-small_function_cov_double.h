#ifndef INTERFACE_H_
#define INTERFACE_H_



int randint(int a);
//void get_img(unsigned char imgout[1120]);
unsigned char get_img(int i);
double get_weight(int a, int i);
void get_convb(double cb[5][28][28]);
void get_densemax(int a, double dmax[10]);

double sigmoid(double x);
int set_sig_layer(double u, int f_dim, int i, int j);

double d_sigmoid(double x);

double softmax_den(double *x, int len);

int initialise_weights();

void forward_pass(unsigned char img[1120]);

int update_weights(int a);

int backward_pass(int a);

int read_train_data();
int read_test_data();

void give_img(unsigned char *vec);
int give_img_train(int randnum);

int give_y(int randnum);
//int conv_sig(int a);
int conv_sig(int a);
int give_prediction();
int filter_ite(unsigned char img1[1120], double weight[49], int filter_dim);
int validate(int i, int cor);

void train(int i);
double convolution(int a, int b, unsigned char img[], int offset_img_a, int offset_img_b, double weight[49]);
int forward_pass_noconv( int a);

#endif
