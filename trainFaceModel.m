function [ model_sample ] = trainFaceModel( tr_images, test_images, tr_labels )
%Returns a struct with a number of trained SVM models (which can be
%modified)
%This function assumes that all test data is passed as "test_images" (i.e.,
%not separate val_images and test_images).

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

testing_label_face = zeros(ntest,1);

%Generate selections of test data (with replacement) for bagging-----------
pointsToPick = 2500;
degreeOfModel = 2;

%Train 5 models by default
for l=1:5
    indices = randperm(2507);
    tr_images_sample = tr_images_face(indices(1:pointsToPick),:);
    tr_labels_sample = tr_labels(indices(1:pointsToPick),:);

    %Following line used for comparing degree of kernel
    %degreeOfModel = degreeOfModel + 0.3;
    
    optionsString = ['-s 0 -c 0.5 -t 0 -g 1 -r 1 -d ' num2str(floor(degreeOfModel))];
    model_sample(l) = svmtrain(tr_labels_sample, tr_images_sample, optionsString);
end

end