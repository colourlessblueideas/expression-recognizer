%Dawson Overton, November 2012

load training.mat;
load val_images.mat;
load test_images.mat;

%if ~exist('test_images', 'var')
  test_images = val_images;
%else
%  test_images = cat(3, val_images, test_images);
%end

%%%Reshape data------------------------------------------------------------

ntr = size(tr_images, 3);
ntest = size(test_images, 3);
h = size(tr_images,1);
w = size(tr_images,2);
tr_images_face = double(reshape(tr_images, [h*w, ntr]));
test_images_face = double(reshape(test_images, [h*w, ntest]));

%%%Subtract mean for each image--------------------------------------------

tr_mu = mean(tr_images_face);
test_mu = mean(test_images_face);
tr_images_face = bsxfun(@minus, tr_images_face, tr_mu);
test_images_face = bsxfun(@minus, test_images_face, test_mu);

%%%Normalize variance for each image---------------------------------------

tr_sd = var(tr_images_face);
tr_sd = tr_sd + 0.01;
tr_sd = sqrt(tr_sd);
tr_images_face = bsxfun(@rdivide, tr_images_face, tr_sd);  

test_sd = var(test_images_face);
test_sd = test_sd + 0.01;
test_sd = sqrt(test_sd);
test_images_face = bsxfun(@rdivide, test_images_face, test_sd);  

tr_images_face = tr_images_face';
test_images_face = test_images_face';


%
test_images_face = tr_images_face(2508:end,:);
tr_images_face = tr_images_face(1:2507,:);
%

%testing_label_face = zeros(ntest,1);
testing_label_face = tr_labels(2508:end);
tr_labels = tr_labels(1:2507);

%======================
%Generate selections of test data (with replacement) for bagging-----------
pointsToPick = 2500;
degreeOfModel = 2;

for l=1:5
    indices = randperm(2507);
    tr_images_sample = tr_images_face(indices(1:pointsToPick),:);
    tr_labels_sample = tr_labels(indices(1:pointsToPick),:);

    %Following line used for comparing degree of kernel
    %degreeOfModel = degreeOfModel + 0.3;
    
    optionsString = ['-s 0 -c 0.5 -t 3 -g 1 -r 1 -d ' num2str(degreeOfModel)];%num2str(floor(degreeOfModel))];
    model_sample(l) = svmtrain(tr_labels_sample, tr_images_sample, optionsString);
    [predict_label_sample(:,l), accuracy_sample(:,l), prob_estimates_sample(:,:,l)] = svmpredict(testing_label_face, test_images_face, model_sample(l), '-b 0');
    fprintf('model %d trained\n',l);
end

%Calculate overall rate correct--------------------------------------------
numCorrectBagging = 0;
for m=1:418
    if mode(predict_label_sample(m,:)) == testing_label_face(m)
        numCorrectBagging = numCorrectBagging + 1;
    end
end

fprintf('total: %d/418 correct',numCorrectBagging);

%======================


% Fill in the test labels with 0 if necessary
%if (length(predict_label_sample) < 1253)
%  predict_label_face = [predict_label_sample; zeros(1253-length(predict_label_sample), 1)];
%end

% Print the predictions to file
fprintf('writing the output to predict_label_sample.csv\n');
fid = fopen('predict_label_sample.csv', 'w');
for i=1:length(predict_label_sample)
  fprintf(fid, '%d\n', predict_label_sample(i));
end
fclose(fid);

clear tr_images test_images val_images