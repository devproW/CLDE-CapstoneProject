package ai.certifai.solution.classification;

import org.apache.commons.io.FilenameUtils;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.*;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingModelSaver;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.TimeUnit;

public class AgeModel {

    private static final Logger log = org.slf4j.LoggerFactory.getLogger(AgeDetection.class);
    private static ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
    private static InputSplit trainData,testData;
    private static DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
    private static ImageTransform transform;
    private static final int height = 224;
    private static final int width = 224;
    private static final int channels = 3;
    private static final int numClasses = 6;
    private static int batchSize = 32;
    private static int seed = 123;
    private static final Random rng = new Random(seed);
    private static final String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
    private static int trainPerc = 80;
    private static int epochs = 20;
    private static File modelFilename = new File(System.getProperty("user.dir"), "generated-models/AgeDetection.zip");




    public static void main(String[] Args) throws Exception
    {

        File dir = new ClassPathResource("Ages").getFile();
        ImageTransform horizontalFlip = new FlipImageTransform(1);
        ImageTransform cropImage = new CropImageTransform(25);
        ImageTransform rotateImage = new RotateImageTransform(rng, 15);
        boolean shuffle = false;
        List<Pair<ImageTransform,Double>> pipeline = Arrays.asList(
                new Pair<>(horizontalFlip,0.5),
                new Pair<>(rotateImage, 0.5),
                new Pair<>(cropImage,0.3));
        transform = new PipelineImageTransform(pipeline,shuffle);
        FileSplit filesInDir = new FileSplit(dir, allowedExtensions, rng);
        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, allowedExtensions, labelMaker);
        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, trainPerc, 100-trainPerc);
        trainData = filesInDirSplit[0];
        testData = filesInDirSplit[1];

        DataSetIterator trainIter = trainIterator();
        DataSetIterator testIter = testIterator();

        //load vgg16 zoo model
//        ZooModel zooModel = AlexNet.builder().build();
//        ComputationGraph alexNet = (ComputationGraph) zooModel.initPretrained(PretrainedType.MNIST);
////        MultiLayerConfiguration conf = ((AlexNet) zooModel).conf();
//        log.info(alexNet.summary());
//
//        // Override the setting for all layers that are not "frozen".
//        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
//                .updater(new Nesterovs(5e-4, 0.9))
//                .seed(seed)
//                .build();
//
//        //Construct a new model with the intended architecture and print summary
//        ComputationGraph alexNetTransfer = new TransferLearning.GraphBuilder(alexNet)
//                .fineTuneConfiguration(fineTuneConf)
//                .setFeatureExtractor("flatten_1") //the specified layer and below are "frozen"
//                .removeVertexKeepConnections("fc1000") //replace the functionality of the final vertex
//                .addLayer("fc1000",
//                        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
//                                .nIn(2048).nOut(numClasses)
//                                .weightInit(WeightInit.XAVIER)
//                                .activation(Activation.SOFTMAX).build(),
//                        "flatten_1")
//                .build();
//        log.info(alexNetTransfer.summary());

        String tempDir = System.getProperty("java.io.tmpdir");
        String exampleDirectory = FilenameUtils.concat(tempDir, "DL4JEarlyStoppingExample/");
        File dirFile = new File(exampleDirectory); //We have to create the temp directory or the sample will fail.
        dirFile.mkdir(); // If mkdir fails, it is probably because the directory already exists. Which is fine.

        EarlyStoppingModelSaver saver = new LocalFileModelSaver(exampleDirectory);

        EarlyStoppingConfiguration esConf = new EarlyStoppingConfiguration.Builder()
                .epochTerminationConditions(new MaxEpochsTerminationCondition(50))
                .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(6, TimeUnit.HOURS))
                .scoreCalculator(new DataSetLossCalculator(testIter, true))
                .evaluateEveryNEpochs(1)
                .modelSaver(saver)
                .build();


//
//        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
//                .seed(seed)
//                .weightInit(WeightInit.XAVIER)
//                .updater(new Adam(0.0001))
//                .l2(0.0005)
//                .list()
//                .layer(0,new ConvolutionLayer.Builder()
//                        .kernelSize(7,7)
//                        .stride(4,4)
//                        .nIn(channels)
//                        .nOut(96)
//                        .activation(Activation.RELU)
//                        .build())
//                .layer(1,new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
//                        .kernelSize(3,3)
//                        .stride(2,2)
//                        .build())
//                .layer(2, new LocalResponseNormalization.Builder().build())
//                .layer(3,new ConvolutionLayer.Builder()
//                        .kernelSize(5,5)
//                        .padding(2,2)
//                        .nIn(96)
//                        .nOut(256)
//                        .activation(Activation.RELU)
//                        .build())
//                .layer(4,new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
//                        .kernelSize(3,3)
//                        .stride(2,2)
//                        .build())
//                .layer(5, new LocalResponseNormalization.Builder().build())
//                .layer(6,new ConvolutionLayer.Builder()
//                        .kernelSize(3,3)
//                        .padding(1,1)
//                        .nIn(256)
//                        .nOut(384)
//                        .activation(Activation.RELU)
//                        .build())
//                .layer(7,new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
//                        .kernelSize(3,3)
//                        .stride(2,2)
//                        .build())
//                .layer(8,new DenseLayer.Builder().activation(Activation.RELU)
//                        .nOut(512)
//                        .dropOut(0.5)
//                        .build())
//                .layer(9,new DenseLayer.Builder()
//                        .nIn(512)
//                        .dropOut(0.5)
//                        .nOut(512)
//                        .activation(Activation.RELU)
//                        .build())
//                .layer(10,new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
//                        .nIn(512)
//                        .nOut(numClasses)
//                        .activation(Activation.SOFTMAX)
//                        .build())
//                .setInputType(InputType.convolutionalFlat(height, width, channels)) // InputType.convolutional for normal image
//                .backpropType(BackpropType.Standard)
//                .build();

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.0001))
                .l2(0.0005)
                .list()
                .layer(0,new BatchNormalization())
                .layer(1,new ConvolutionLayer.Builder()
                        .kernelSize(5,5)
                        .stride(2,2)
                        .nIn(channels)
                        .nOut(100)
                        .activation(Activation.RELU)
                        .build())
                .layer(2,new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(1,1)
                        .build())
                .layer(3,new ConvolutionLayer.Builder()
                        .kernelSize(5,5)
                        .stride(2,2)
                        .nOut(100)
                        .activation(Activation.RELU)
                        .build())
                .layer(4,new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(1,1)
                        .build())
                .layer(5, new DenseLayer.Builder()
                        .nOut(100)
                        .activation(Activation.RELU)
                        .build()).layer(6,new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nIn(100)
                        .nOut(numClasses)
                        .build())
                .setInputType(InputType.convolutional(height,width,channels))
                .backpropType(BackpropType.Standard)
                .build();


        StatsStorage storage = new InMemoryStatsStorage();
        UIServer server = UIServer.getInstance();
        server.attach(storage);

//        if(esConf==null) {
//            System.out.println("Configuration not found!");
//            System.exit(0);
//        }
//
        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf, conf, trainIter);

        //Conduct early stopping training:
        EarlyStoppingResult result = trainer.fit();
        System.out.println("Termination reason: " + result.getTerminationReason());
        System.out.println("Termination details: " + result.getTerminationDetails());
        System.out.println("Total epochs: " + result.getTotalEpochs());
        System.out.println("Best epoch number: " + result.getBestModelEpoch());
        System.out.println("Score at best epoch: " + result.getBestModelScore());

        //Print score vs. epoch
        Map<Integer, Double> scoreVsEpoch = result.getScoreVsEpoch();
        List<Integer> list = new ArrayList<>(scoreVsEpoch.keySet());
        Collections.sort(list);
        System.out.println("Score vs. Epoch:");
        for (Integer i : list) {
            System.out.println(i + "\t" + scoreVsEpoch.get(i));
        }

//        MultiLayerNetwork model = new MultiLayerNetwork(conf);
//        model.init();
//        model.setListeners(new ScoreIterationListener(10));
//
//        model.fit(trainIter,epochs);
//        {
//            Evaluation evaluation = model.evaluate(testIter);
//            System.out.println(evaluation.stats());
//        }
//        {
//            Evaluation evaluation = model.evaluate(trainIter);
//            System.out.println(evaluation.stats());
//        }

//        alexNetTransfer.setListeners(
//                new StatsListener( storage),
//                new ScoreIterationListener(5),
//                new EvaluativeListener(trainIter, 1, InvocationType.EPOCH_END),
//                new EvaluativeListener(testIter, 1, InvocationType.EPOCH_END)
//        );

//        alexNetTransfer.fit(trainIter, 1);

        MultiLayerNetwork model = (MultiLayerNetwork)result.getBestModel();

        trainIter.reset();
        testIter.reset();
        {
            Evaluation evaluation = model.evaluate(trainIter);
            System.out.println(evaluation.stats());
        }
        {
            Evaluation evaluation = model.evaluate(testIter);
            System.out.println(evaluation.stats());
        }

        ModelSerializer.writeModel(model, modelFilename, true);

    }

    private static DataSetIterator makeIterator(InputSplit split, boolean training) throws IOException {
        ImageRecordReader recordReader = new ImageRecordReader(height,width,channels,labelMaker);
        if (training && transform != null){
            recordReader.initialize(split,transform);
        }else{
            recordReader.initialize(split);
        }
        DataSetIterator iter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numClasses);
        iter.setPreProcessor(scaler);

        return iter;
    }

    public static DataSetIterator trainIterator() throws IOException {
        return makeIterator(trainData, true);
    }

    public static DataSetIterator testIterator() throws IOException {
        return makeIterator(testData, false);
    }

}
