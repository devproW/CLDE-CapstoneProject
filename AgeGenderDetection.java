package ai.certifai.solution.facial_recognition;

import ai.certifai.solution.classification.HaarFaceDetector;
import ai.certifai.solution.classification.ImageUtils;
import ai.certifai.solution.facial_recognition.identification.Prediction;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacv.OpenCVFrameGrabber;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_core.Point;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.swing.*;
import java.awt.*;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;

public class AgeGenderDetection {
    private static final Logger logger = LoggerFactory.getLogger(FaceRecognitionWebcam.class);
    private static final int WIDTH = 1280;
    private static final int HEIGHT = 720;
    private static final String outputWindowsName = "Face Recognition Example - DL4J";
    private static File CNNAgeModel = new File(System.getProperty("user.dir"), "generated-models/AgeDetection.zip");
    private static File CNNGenderModel = new File(System.getProperty("user.dir"), "generated-models/GenderDetection.zip");
    private static MultiLayerNetwork AgeModel;
    private static MultiLayerNetwork GenderModel;
    private static final int height = 224;
    private static final int width = 224;
    private static final int channels = 3;

    private static List<Prediction> predictions;
    private static final String[] AGES = new String[]{"0-7", "15-24", "25-32", "33-47", "48-59", "60-", "8-14"};

    private FrameGrabber frameGrabber;
    private OpenCVFrameConverter.ToMat toMatConverter = new OpenCVFrameConverter.ToMat();
    private volatile boolean running = false;

    private HaarFaceDetector faceDetector = new HaarFaceDetector();

    private JFrame window;
    private JPanel videoPanel;

    public static void main(String[] args) throws Exception {

        if (new File(CNNAgeModel.toString()).exists()) {
            logger.info("Load model...");
            AgeModel = ModelSerializer.restoreMultiLayerNetwork(CNNAgeModel);
            logger.info("Model found.");
        } else {
            logger.info("Model not found.");
        }

        if (new File(CNNGenderModel.toString()).exists()) {
            logger.info("Load model...");
            GenderModel = ModelSerializer.restoreMultiLayerNetwork(CNNGenderModel);
            logger.info("Model found.");
        } else {
            logger.info("Model not found.");
        }

        AgeDetection javaCVExample = new AgeDetection();

        logger.info("Starting javacv example");
        new Thread(javaCVExample::start).start();

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            logger.info("Stopping javacv example");
            javaCVExample.stop();
        }));

        try {
            Thread.currentThread().join();
        } catch (InterruptedException ignored) { }
    }

    public AgeGenderDetection() {
        window = new JFrame();
        videoPanel = new JPanel();

        window.setLayout(new BorderLayout());
        window.setSize(new Dimension(1280, 720));
        window.add(videoPanel, BorderLayout.CENTER);
        window.addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                stop();
            }
        });
    }
    private void process() {
        running = true;
        while (running) {
            try {
                // Here we grab frames from our camera
                final org.bytedeco.javacv.Frame frame = frameGrabber.grab();

                Map<Rect, Mat> detectedFaces = faceDetector.detect(frame);
                Mat mat = toMatConverter.convert(frame);

                detectedFaces.entrySet().forEach(rectMatEntry -> {
                    String age = predictAge(rectMatEntry.getValue(), frame);
                    AgeDetection.Gender gender = predictGender(rectMatEntry.getValue(), frame);
                    String caption = String.format("%s:[%s]",gender, age);
                    logger.debug("Face's caption : {}", caption);

                    rectangle(mat, new org.bytedeco.opencv.opencv_core.Point(rectMatEntry.getKey().x(), rectMatEntry.getKey().y()),
                            new org.bytedeco.opencv.opencv_core.Point(rectMatEntry.getKey().width() + rectMatEntry.getKey().x(), rectMatEntry.getKey().height() + rectMatEntry.getKey().y()),
                            Scalar.RED, 2, CV_AA, 0);

                    int posX = Math.max(rectMatEntry.getKey().x() - 10, 0);
                    int posY = Math.max(rectMatEntry.getKey().y() - 10, 0);
                    putText(mat, caption, new Point(posX, posY), CV_FONT_HERSHEY_PLAIN, 1.0,
                            new Scalar(255, 255, 255, 2.0));
                });

                // Show the processed mat in UI
                org.bytedeco.javacv.Frame processedFrame = toMatConverter.convert(mat);

                Graphics graphics = videoPanel.getGraphics();
                BufferedImage resizedImage = ImageUtils.getResizedBufferedImage(processedFrame, videoPanel);
                SwingUtilities.invokeLater(() -> {
                    graphics.drawImage(resizedImage, 0, 0, videoPanel);
                });
            } catch (FrameGrabber.Exception e) {
                logger.error("Error when grabbing the frame", e);
            } catch (Exception e) {
                logger.error("Unexpected error occurred while grabbing and processing a frame", e);
            }
        }
    }
    public void start() {
        // frameGrabber = new FFmpegFrameGrabber("/dev/video0");
        // The available FrameGrabber classes include OpenCVFrameGrabber (opencv_videoio),
        // DC1394FrameGrabber, FlyCapture2FrameGrabber, OpenKinectFrameGrabber,
        // PS3EyeFrameGrabber, VideoInputFrameGrabber, and FFmpegFrameGrabber.
        frameGrabber = new OpenCVFrameGrabber(0);

        //frameGrabber.setFormat("mp4");
        frameGrabber.setImageWidth(1280);
        frameGrabber.setImageHeight(720);

        logger.debug("Starting frame grabber");
        try {
            frameGrabber.start();
            logger.debug("Started frame grabber with image width-height : {}-{}", frameGrabber.getImageWidth(), frameGrabber.getImageHeight());
        } catch (FrameGrabber.Exception e) {
            logger.error("Error when initializing the frame grabber", e);
            throw new RuntimeException("Unable to start the FrameGrabber", e);
        }

        SwingUtilities.invokeLater(() -> {
            window.setVisible(true);
        });

        process();

        logger.debug("Stopped frame grabbing.");
    }
    public void stop() {
        running = false;
        try {
            logger.debug("Releasing and stopping FrameGrabber");
            frameGrabber.release();
            frameGrabber.stop();
        } catch (FrameGrabber.Exception e) {
            logger.error("Error occurred when stopping the FrameGrabber", e);
        }

        window.dispose();
    }

    public String predictAge(Mat face, org.bytedeco.javacv.Frame frame) {
        try {
            resize(face, face, new Size(224, 224));
            //ageNet.setInput(inputBlob, "data", 1.0, null);      //set the network input

            NativeImageLoader loader = new NativeImageLoader();

//                resize(inputBlob,inputBlob,new Size(224,224));

            INDArray ds = null;
            try {
                ds = loader.asMatrix(face);
            } catch (IOException ex) {
                logger.error(ex.getMessage());
            }

            ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
            scaler.transform(ds);
            System.out.println(java.util.Arrays.toString(ds.shape()));
            INDArray results = AgeModel.output(ds);
            INDArray t = Nd4j.argMax(results,1);
            int c = t.getInt(0);


            return AGES[c];
        } catch (Exception e) {
            logger.error("Error when processing gender", e);
        }
        return null;
    }

    public AgeDetection.Gender predictGender(Mat face, Frame frame) {
        try {
            resize(face, face, new Size(224, 224));

            NativeImageLoader loader = new NativeImageLoader();

//            resize(inputBlob,inputBlob,new Size(224,224));

            INDArray ds = null;
            try {
                ds = loader.asMatrix(face);
            } catch (IOException ex) {
                logger.error(ex.getMessage());
            }

            ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
            scaler.transform(ds);
            System.out.println(Arrays.toString(ds.shape()));
            INDArray results = GenderModel.output(ds);
            double a = results.getDouble(0,0);
            double b = results.getDouble(0,1);
            logger.debug("CNN results {},{}", results.getScalar(0,0), results.getScalar(0,1));

            if (a > b) {
                logger.debug("Male detected");
                return AgeDetection.Gender.MALE;
            } else {
                logger.debug("Female detected");
                return AgeDetection.Gender.FEMALE;
            }
        } catch (Exception e) {
            logger.error("Error when processing gender", e);
        }
        return AgeDetection.Gender.NOT_RECOGNIZED;
    }

    public enum Gender {
        MALE,
        FEMALE,
        NOT_RECOGNIZED
    }


//        //        STEP 1 : Select your face detector and face identifier
//        FaceDetector FaceDetector = getFaceDetector(ai.certifai.solution.facial_recognition.detection.FaceDetector.OPENCV_HAAR_CASCADE_FACEDETECTOR);
//        FaceIdentifier FaceIdentifier = getFaceIdentifier(ai.certifai.solution.facial_recognition.identification.FaceIdentifier.FEATURE_DISTANCE_INCEPTION_RESNET_PREBUILT);
//
//        //        STEP 2 : Stream the video frame from camera
//        VideoCapture capture = new VideoCapture();
//        capture.set(CAP_PROP_FRAME_WIDTH, WIDTH);
//        capture.set(CAP_PROP_FRAME_HEIGHT, HEIGHT);
//        namedWindow(outputWindowsName, WINDOW_NORMAL);
//        resizeWindow(outputWindowsName, 1280, 720);
//
//
//        if (!capture.open(0)) {
//            System.out.println("Cannot open the camera !!!");
//        }
//
//        Mat image = new Mat();
//        Mat cloneCopy = new Mat();
//
//        while (capture.read(image)) {
//            flip(image, image, 1);
//
//            //        STEP 3 : Perform face detection
//            image.copyTo(cloneCopy);
//            FaceDetector.detectFaces(cloneCopy);
//            List<FaceLocalization> faceLocalizations = FaceDetector.getFaceLocalization();
//            annotateFaces(faceLocalizations, image);
//
//            //        STEP 4 : Perform face recognition
//            image.copyTo(cloneCopy);
//            detect(cloneCopy);
//            List<List<ai.certifai.solution.facial_recognition.identification.Prediction>> faceIdentifier = FaceIdentifier.recognize(faceLocalizations,cloneCopy);
//            labelIndividual(faceIdentifier,image);
//
//            //        STEP 5 : Display output in a window
//            imshow(outputWindowsName, image);
//
//            char key = (char) waitKey(20);
//            // Exit this loop on escape:
//            if (key == 27) {
//                destroyAllWindows();
//                break;
//            }
//


//        private static void annotateFaces(List<FaceLocalization> faceLocalizations, Mat image) {
//            for (FaceLocalization i : faceLocalizations){
//                rectangle(image,new Rect(new Point((int) i.getLeft_x(),(int) i.getLeft_y()), new Point((int) i.getRight_x(),(int) i.getRight_y())), new Scalar(0, 255, 0, 0),2,8,0);
//            }
//        }
//
//    private static FaceDetector getFaceDetector(String faceDetector) throws IOException {
//        switch (faceDetector) {
//            case FaceDetector.OPENCV_HAAR_CASCADE_FACEDETECTOR:
//                return new OpenCV_HaarCascadeFaceDetector();
//            case FaceDetector.OPENCV_DL_FACEDETECTOR:
//                return new OpenCV_DeepLearningFaceDetector(300, 300, 0.8);
//            default:
//                return  null;
//        }
//    }
//
//    private static void detect(Mat image)throws InterruptedException, IOException{
//        NativeImageLoader loader = new NativeImageLoader();
//        resize(image,image,new Size(224,224));
//        INDArray ds = null;
//        try {
//            ds = loader.asMatrix(image);
//        } catch (IOException ex) {
//            log.error(ex.getMessage());
//        }
//
//        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
//        scaler.transform(ds);
//        System.out.println(Arrays.toString(ds.shape()));
//        INDArray results = model.output(ds);
//        getPredictions(results);
//    }
//
//    public static void getPredictions(INDArray result) throws IOException {
//        predictions = decodePredictions(result, 3);
//        System.out.println("prediction: ");
//        System.out.println(predictionsToString(predictions));
//    }
//
//    private static String predictionsToString(List<ai.certifai.solution.facial_recognition.identification.Prediction> predictions) {
//        StringBuilder builder = new StringBuilder();
//        for (ai.certifai.solution.facial_recognition.identification.Prediction prediction : predictions) {
//            builder.append(prediction.toString());
//            builder.append('\n');
//        }
//        return builder.toString();
//    }
//
//    private static List<ai.certifai.solution.facial_recognition.identification.Prediction> decodePredictions(INDArray encodedPredictions, int numPredicted) throws IOException {
//        List<ai.certifai.solution.facial_recognition.identification.Prediction> decodedPredictions = new ArrayList<>();
//        int[] topX = new int[numPredicted];
//        float[] topXProb = new float[numPredicted];
//
//        int i = 0;
//        for (INDArray currentBatch = encodedPredictions.getRow(0).dup(); i < numPredicted; ++i) {
//
//            topX[i] = Nd4j.argMax(currentBatch, 1).getInt(0);
//            topXProb[i] = currentBatch.getFloat(0, topX[i]);
//            currentBatch.putScalar(0, topX[i], 0.0D);
//            decodedPredictions.add(new ai.certifai.solution.facial_recognition.identification.Prediction(labels.get(topX[i]), (topXProb[i] * 100.0F)));
//        }
//        return decodedPredictions;
//    }
//
//
//    private static void labelIndividual(List<List<ai.certifai.solution.facial_recognition.identification.Prediction>> decodePrediction, Mat Image){
//        for (List<ai.certifai.solution.facial_recognition.identification.Prediction> i: decodePrediction){
//            for(int j=0; j<i.size(); j++)
//            {
//                putText(
//                        Image,
//                        i.get(j).toString(),
//                        new Point(
//                                (int)i.get(j).getFaceLocalization().getLeft_x() + 2,
//                                (int)i.get(j).getFaceLocalization().getLeft_y() - 5
//                        ),
//                        FONT_HERSHEY_COMPLEX,
//                        0.5,
//                        Scalar.YELLOW
//                );
//            }
//        }
//    }
//
//    private static List<List<ai.certifai.solution.facial_recognition.identification.Prediction>> recognize(List<FaceLocalization> faceLocalizations, Mat image) throws IOException {
//        return null;
//    }
//
//    public static class Prediction {
//
//        private String label;
//        private double percentage;
//
//        public Prediction(String label, double percentage) {
//            this.label = label;
//            this.percentage = percentage;
//        }
//
//        public void setLabel(final String label) {
//            this.label = label;
//        }
//
//        public String toString() {
//            return String.format("%s: %.2f ", this.label, this.percentage);
//        }
//    }
//
//    private static FaceIdentifier getFaceIdentifier(String faceIdentifier) throws IOException, ClassNotFoundException {
//        switch (faceIdentifier) {
//            case FaceIdentifier.FEATURE_DISTANCE_VGG16_PREBUILT:
//                return new DistanceFaceIdentifier(
//                        new VGG16FeatureProvider(),
//                        new ClassPathResource("FaceDB").getFile(), 0.3, 3);
//            case FaceIdentifier.FEATURE_DISTANCE_INCEPTION_RESNET_PREBUILT:
//                return new DistanceFaceIdentifier(
//                        new InceptionResNetFeatureProvider(),
//                        new ClassPathResource("FaceDB").getFile(), 0.3, 3);
//            default:
//                return null;
//        }
//    }


}
