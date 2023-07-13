#ifndef INTERFACE_H_
#define INTERFACE_H_

double sigmoid(double x); 

double d_sigmoid(double x);

double softmax_den(double *x, int len);

void initialise_weights();

void forward_pass(unsigned char img[][32]);

void update_weights();

void backward_pass(double *y_hat, int *y, unsigned char img[][32]); 

void read_train_data(); 

void read_test_data();

void give_img(unsigned char* vec , unsigned char img[][32]);

void give_y(int y, int *vector_y);

int give_prediction();

double convolution(int a, int b, unsigned char *img, int offset_img_a, int offset_img_b, double *weights);

#endif
