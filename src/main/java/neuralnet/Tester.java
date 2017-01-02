package neuralnet;

import org.apache.commons.io.FileUtils;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.examples.feedforward.regression.function.MathFunction;
import org.deeplearning4j.examples.recurrent.seq2seq.CustomSequenceIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.graph.rnn.DuplicateToTimeSeriesVertex;
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.RefineryUtilities;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;

import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.WindowConstants;

public class Tester {
	
	// Hyper-parameters
	
	public static final int miniBatchSize = 1;
	public static final int numPossibleLabels = -1;
	public static final int labelIndex = 1; // Zero indexed
	
	public static final int lstmLayerSize = 400;
	public static final int tbpttLength = 66; 
	
	public static final int numEpochs = 30;
	
    public static void main(String[] args) throws Exception {
    	
    	SequenceRecordReader reader = new CSVSequenceRecordReader(0, ",");
    	reader.initialize(new NumberedFileInputSplit("C:/Users/Niall/Desktop/NeuralNet/quotes_%d.csv", 0, 0));
    	DataSetIterator iter = new SequenceRecordReaderDataSetIterator(reader, miniBatchSize, numPossibleLabels, labelIndex, true);
    	
    	
    	SequenceRecordReader reader2 = new CSVSequenceRecordReader(0, ",");
    	reader2.initialize(new NumberedFileInputSplit("C:/Users/Niall/Desktop/NeuralNet/quotes_%d.csv", 1, 1));
    	DataSetIterator iter2 = new SequenceRecordReaderDataSetIterator(reader2, miniBatchSize, numPossibleLabels, labelIndex, true);
    	
    	
    	int nOut = iter.inputColumns();
    	
    	MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
    			.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(2)
    			.learningRate(0.01)
    			.rmsDecay(0.95)
    			.seed(12345)
    			.regularization(true)
    			.l2(0.001)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.RMSPROP)
    			.list()
    			.layer(0, new GravesLSTM.Builder().nIn(iter.inputColumns()).nOut(lstmLayerSize)
    					.activation("tanh").build())
    			.layer(1, new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
    					.activation("tanh").build())
    			.layer(2, new RnnOutputLayer.Builder(LossFunction.MSE).activation("identity")
    					.nIn(lstmLayerSize).nOut(nOut).build())
                .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
    			.pretrain(false).backprop(true)
    			.build();

		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();
		net.setListeners(new ScoreIterationListener(1));
		
		Layer[] layers = net.getLayers();
		int totalNumParams = 0;
		
		for( int i = 0; i < layers.length; i++){
			int nParams = layers[i].numParams();
			System.out.println("Number of parameters in layer " + i + ": " + nParams);
			totalNumParams += nParams;
		}
		
		System.out.println("Total number of network parameters: " + totalNumParams);
		
        for( int i = 0; i < numEpochs; i++) {
        	net.fit(iter);
        	iter.reset();
        	
        	RegressionEvaluation eval = new RegressionEvaluation(1);
        	
        	while (iter2.hasNext()) {
        		DataSet ds = iter2.next();
        		INDArray features = ds.getFeatureMatrix();
                INDArray labels = ds.getLabels();
                INDArray predicted = net.output(features,false);
                System.out.println(predicted);
                eval.evalTimeSeries(labels,predicted);
                
                System.out.println(String.format("Standardized Prediction:  %s, Standardized Actual: %s", 
                        predicted.toString(), labels.toString()));
        	}
        	
        	//System.out.println(eval.stats());

            iter2.reset();
        }
        
        while (iter.hasNext()) {
            DataSet t = iter.next();
            net.rnnTimeStep(t.getFeatureMatrix());
        }

        iter.reset();
        
		System.out.println("\n\nFinished");
		
    }
    
}

