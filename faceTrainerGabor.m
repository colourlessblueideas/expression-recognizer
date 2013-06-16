%Dawson Overton, November 2012

load training.mat;
load val_images.mat;
load test_images.mat;

if ~exist('test_images', 'var')
  test_images = val_images;
else
  test_images = cat(3, val_images, test_images);
end

ntr = size(tr_images, 3);
ntest = size(test_images, 3);
h = size(tr_images,1);
w = size(tr_images,2);

%%%Gabor filter------------------------------------------------------------

%Filter parameters
lambda  = 3;
theta   = 0;
psi     = [0 pi/2];
gamma   = 2.5;
bw      = 0.9;
N       = 8;

for m=1:ntr
    img_in = tr_images(:,:,m);
    img_out = zeros(size(img_in,1), size(img_in,2), N);

    for n=1:N
    gb = gabor_fn(bw,gamma,psi(1),lambda,theta)...
        + 1i * gabor_fn(bw,gamma,psi(2),lambda,theta);
    % gb is the n-th gabor filter
    img_out(:,:,n) = imfilter(img_in, gb, 'symmetric');
    % filter output to the n-th channel
    theta = theta + 2*pi/N;
    % next orientation
    end

img_out_disp = sum(abs(img_out).^2, 3).^0.5;
% default superposition method, L2-norm
img_out_disp = img_out_disp./max(img_out_disp(:));

tr_images(:,:,m) = img_out_disp;
fprintf('Gabor filter complete on image %d\n', m);
end

%%%Data reshaping----------------------------------------------------------

tr_images_face = double(reshape(tr_images, [h*w, ntr]));
test_images_face = double(reshape(test_images, [h*w, ntest]));

tr_images_face = tr_images_face';
test_images_face = test_images_face';

%%%------------------------------------------------------------------------

%Next two lines used for personal testing
test_images_face = tr_images_face(2508:end,:);
tr_images_face = tr_images_face(1:2507,:);

%testing_label_face = zeros(ntest,1);
%Next two lines used for personal testing
testing_label_face = tr_labels(2508:end);
tr_labels = tr_labels(1:2507);

%Train SVM model on
model_face = svmtrain(tr_labels, tr_images_face, '-s 0 -c 1 -t 1 -g 0.25 -r 1 -d 2');

%Predict labels
[predict_label_face, accuracy_face, prob_estimates_face] = svmpredict(testing_label_face, test_images_face, model_face, '-b 0');

% Fill in the test labels with 0 if necessary
if (length(predict_label_face) < 1253)
  predict_label_face = [predict_label_face; zeros(1253-length(predict_label_face), 1)];
end

% Print the predictions to file
fprintf('writing the output to predict_label_gabor.csv\n');
fid = fopen('predict_label_gabor.csv', 'w');
for i=1:length(predict_label_face)
  fprintf(fid, '%d\n', predict_label_face(i));
end
fclose(fid);

%Cleanup
clear tr_images test_images val_images