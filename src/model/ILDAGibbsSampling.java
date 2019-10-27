package model;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.math3.random.JDKRandomGenerator;

import model.Perplexity;
import util.FileUtil;
import util.FuncUtils;


/**
 * Sparse Background Topic Model--GibbsSampling
 * 
 * @author: qianyang
 * @email qy20115549@126.com
 */
public class ILDAGibbsSampling {

	public Map<String, Integer> wordToIndexMap = new HashMap<String, Integer>();;  //word to index
	public List<String> indexToWordMap = new ArrayList<String>();    //index to String word 
	public int[][] docword;
	public int M;  //document size
	public int V; // number of words in the corpus
	public int K;
	public double alpha ;  //超参数 alpha0 = 1E-12
	public double beta; 
	public double beta_back;
	public double gamma0;  //for beta distribution 
	double gamma1; //for beta distribution 
	int[][] z;
	public int[][] ndk; // document-topic count
	public int[] ndsum; //document-topic sum
	public int[][] nkw; //topic-word count
	public int[] nksum; //topic-word sum (total number of words assigned to a topic)
	boolean c[][]; //背景词选择器
	public long[] n_cv; //2维的  背景词0对应的总单词数量  和 非背景词1对应的总单词数量
	public int[] nback_v; //1*V 背景词某单词的频率
	JDKRandomGenerator rand; //随机数生成器
	//output
	public int topWordsOutputNumber;
	public String outputFileDirectory; 
	public String code; 
	int iterations;
	public int topWords; // number of most probable words for each topic
	public ILDAGibbsSampling(String inputFile, String inputFileCode, int topicNumber,
	 double inputAlpha, double inputBeta, double inputBeta_back,double inputGamma0, double inputGamma1, int inputIterations, int inTopWords,
			String outputFileDir){
		//read data
		ArrayList<String> docLines = new ArrayList<String>();
		FileUtil.readLines(inputFile, docLines,inputFileCode);
		M = docLines.size();
		docword = new int[M][];
		int j = 0;
		for(String line : docLines){
			List<String> words = new ArrayList<String>();
			FileUtil.tokenizeAndLowerCase(line, words,"\\s+");
			docword[j] = new int[words.size()];
			for(int i = 0; i < words.size(); i++){
				String word = words.get(i);
				if(!wordToIndexMap.containsKey(word)){
					int newIndex = wordToIndexMap.size();
					wordToIndexMap.put(word, newIndex);
					indexToWordMap.add(word);
					docword[j][i] = newIndex;
				} else {
					docword[j][i] = wordToIndexMap.get(word);
				}
			}
			j++;

		}
		V = indexToWordMap.size();
		alpha = inputAlpha;
		beta = inputBeta;
		beta_back = inputBeta_back;
		gamma0 = inputGamma0;
		gamma1 = inputGamma1;
		K = topicNumber;
		iterations = inputIterations;
		topWordsOutputNumber = inTopWords;
		outputFileDirectory = outputFileDir;
		code = inputFileCode;
		initialize();
	}
	//initialize the model
	public void initialize() {
		rand = new JDKRandomGenerator();
		rand.setSeed(System.currentTimeMillis());
		//文档d中主题k生成的单词数目
		ndk = new int[M][K];
		//每篇文档包含的所有单词总数
		ndsum = new int[M];
		//主题k中单词v的数目
		nkw = new int[K][V];
		//主题k对应的单词总数
		nksum = new int[K];
		//每篇文档单词对应的主题
		z = new int[M][];
		//某词是否为背景词0-1初始化
		c = new boolean[M][];
		//背景词和非背景词总数统计
		n_cv = new long[2];
		//数量统计 某词属于背景词
		nback_v = new int[V];
		// assign topics
		for (int d = 0; d < M; d++) {
			// words
			int Nd = docword[d].length;
			z[d] = new int[Nd];
			//循环每一个单词
			for (int n = 0; n < Nd; n++) {
				//随机赋值主题
				int topic = (int) (Math.random() * K);
				z[d][n] = topic;
			}
		}
		//assign label c
		for (int d = 0; d < M; d++) {
			c[d] = new boolean[docword[d].length];
			for (int n = 0; n < docword[d].length; n++) {
				if (Math.random() > 0.5) {
					c[d][n] = true;  //true表示不是背景词
					//更新统计
					updateCount(d, z[d][n], docword[d][n], +1);
				} else {   //如果是背景词
					c[d][n] = false;
					updateCountBackWord(docword[d][n], 1);
				}

			}
		}
	}
	public void MCMCSampling() {
		//抽样更新
		for (int i = 0; i < this.iterations; i++) {
			System.out.println("iteration : " + i);
			gibbsOneIteration();  //执行gibbs采样
		}
		System.out.println("write topic word ..." );
		writeTopWordsWithProbability();
		System.out.println("write background topic word ..." );
		writeTopWordsWithProbability_Bar();
		//writeTopWords();
		System.out.println("write document topic ..." );
		writeDocumentTopic();
		System.out.println("write perplexity ..." );
		writePerplexity();
	}
	//gibbs采样
	public void gibbsOneIteration() {
		//sample topic
		for (int d = 0; d < docword.length; d++) {
			//循环所有单词
			for (int n = 0; n < z[d].length; n++) {
				int topic = sampleFullConditional(d, n);
				//先抽取主题
				z[d][n] = topic;
				if (c[d][n]) {  //表示不是背景词
					ndk[d][topic] += 1;
					//文档d
					ndsum[d] += 1;
					//主题topic对应的单词word数量加1
					nkw[topic][docword[d][n]] += 1;
					//主题topic对应的单词总数加1
					nksum[topic] += 1;
				}
			}
		}
		for (int d = 0; d < docword.length; d++) {
			//循环所有单词
			for (int n = 0; n < c[d].length; n++) {
				sample_label(d, n);
			}
		}
		//重新统计数目
		cleanTempPrmts();
		for (int d = 0; d < docword.length; d++) {
			//循环所有单词
			for (int n = 0; n < c[d].length; n++) {
				if (c[d][n]) {
					//更新统计
					updateCount(d, z[d][n], docword[d][n], +1);
				}else {
					updateCountBackWord(docword[d][n], 1);
				}
			}
		}
	}
	//针对不为背景词的单词抽取主题
	int sampleFullConditional(int d, int n) {
		//获取原对应的主题
		int topic = z[d][n];
		if (c[d][n]) { //如果不为背景词
			ndk[d][topic] += -1;
			//文档d
			ndsum[d] += -1;
			//主题topic对应的单词word数量加1
			nkw[topic][docword[d][n]] += -1;
			//主题topic对应的单词总数加1
			nksum[topic] += -1;
		}
		//概率
		double[] p = new double[K];
		//循环每个主题
		for (int k = 0; k < K; k++) {
			p[k] = (ndk[d][k] + alpha) / (ndsum[d] + K * alpha) * (nkw[k][docword[d][n]] + beta)
					/ (nksum[k] + V * beta);
		}
		//轮盘赌抽取新主题
		topic = sample(p);
		//返回主题
		return topic;

	}
	private void sample_label(int d, int n) {
		boolean binarylabel = c[d][n];
		int binary;
		if (binarylabel == true) {
			binary = 1;
		} else {
			binary = 0;
		}
		n_cv[binary]--;
		if (binary == 0) {  //如果是背景词
			nback_v[docword[d][n]]--;
		} else {   //如果不是背景词
			ndk[d][z[d][n]]--;
			//文档d
			ndsum[d]--;
			//主题topic对应的单词word数量加1
			nkw[z[d][n]][docword[d][n]]--;
			//主题topic对应的单词总数加1
			nksum[z[d][n]]--;
		}
		binarylabel = draw_label(d, n);
		c[d][n] = binarylabel;
	}
	public void cleanTempPrmts() {
		ndk = new int[M][K];
		ndsum = new int[M];
		//主题k中单词v的数目
		nkw = new int[K][V];
		//主题k对应的单词总数
		nksum = new int[K];
		//每篇文档单词对应的主题
		n_cv = new long[2];
		//数量统计 某词属于背景词
		nback_v = new int[V];
	}
	private boolean draw_label(int d, int n) {
		boolean returnvalue = false;
		double[] P_lv;
		P_lv = new double[2];
		double Pb = 1;
		double Ptopic = 1;

		P_lv[0] = (n_cv[0] + gamma0)
				/ (n_cv[0] + n_cv[1] + gamma0 + gamma1); // part 1 from

		P_lv[1] = (n_cv[1] + gamma1)
				/ (n_cv[0] + n_cv[1] + gamma0 + gamma1);

		Pb = (nback_v[docword[d][n]] + beta_back)
				/ (n_cv[0] + V*beta_back); // word in background part(2)
		Ptopic = (nkw[z[d][n]][docword[d][n]] + beta)
				/ (nksum[z[d][n]] + V*beta);

		double p0 = Pb * P_lv[0];
		double p1 = Ptopic * P_lv[1];

		double sum = p0 + p1;
		double randPick = Math.random();

		if (randPick <= p0 / sum) {
			returnvalue = false;
		} else {
			returnvalue = true;
		}
		return returnvalue;
	}
	//轮盘赌
	int sample(double[] p) {

		int topic = 0;
		for (int k = 1; k < p.length; k++) {
			p[k] += p[k - 1];
		}
		double u = Math.random() * p[p.length - 1];
		for (int t = 0; t < p.length; t++) {
			if (u < p[t]) {
				topic = t;
				break;
			}
		}
		return topic;
	}
	//更新统计
	void updateCount(int d, int topic, int word, int flag) {
		//文档 d中的主题topic对应的单词数目加1
		ndk[d][topic] += flag;
		//文档d
		ndsum[d] += flag;
		//主题topic对应的单词word数量加1
		nkw[topic][word] += flag;
		//主题topic对应的单词总数加1
		nksum[topic] += flag;
		//不是背景词的单词数目
		n_cv[1] += flag; 
	}
	void updateCount_New(int d, int topic, int word, int flag) {
		//文档 d中的主题topic对应的单词数目加1
		ndk[d][topic] += flag;
		//文档d
		ndsum[d] += flag;
		//主题topic对应的单词word数量加1
		nkw[topic][word] += flag;
		//主题topic对应的单词总数加1
		nksum[topic] += flag;
		//不是背景词的单词数目
		n_cv[1] += flag; 
	}
	//更新统计  背景词
	void updateCountBackWord(int word, int flag) {
		nback_v[word] += flag;
		n_cv[0] += flag;  //背景词数目统计
	}
	//估计Theta
	public double[][] estimateTheta() {
		double[][] theta = new double[docword.length][K];
		for (int d = 0; d < docword.length; d++) {
			for (int k = 0; k < K; k++) {
				theta[d][k] = (ndk[d][k] + + alpha) / (ndsum[d] + + K * alpha);
			}
		}
		return theta;
	}
	//估计Phi
	public double[][] estimatePhi() {
		double[][] phi = new double[K][V];
		for (int k = 0; k < K; k++) {
			for (int w = 0; w < V; w++) {
				phi[k][w] = (nkw[k][w] + beta) / (nksum[k] + V * beta);
			}
		}
		return phi;
	}
	//估计Phi
	public double[] estimatePhi_Bar() {
		double[] phi_bar = new double[V];
		for (int w = 0; w < V; w++) {
			phi_bar[w] =  (nback_v[w] + beta_back)
					/ (n_cv[0] + V*beta_back);;
		}
		return phi_bar;
	}
	/**
	 * write top words with probability for each topic
	 */
	public void writeTopWordsWithProbability(){
		StringBuilder sBuilder = new StringBuilder();
		double[][] phi = estimatePhi();
		int topicNumber = 1;
		for (double[] phi_z : phi) {
			sBuilder.append("Topic:" + topicNumber + "\n");
			for (int i = 0; i < topWordsOutputNumber; i++) {
				int max_index = FuncUtils.maxValueIndex(phi_z);
				sBuilder.append(indexToWordMap.get(max_index) + " :" + phi_z[max_index] + "\n");
				phi_z[max_index] = 0;
			}
			sBuilder.append("\n");
			topicNumber++;
		}
		try {
			FileUtil.writeFile(outputFileDirectory + "ILDA_topic_word_" + K + ".txt", sBuilder.toString(),code);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	/**
	 * write top words with probability for background  topic
	 */
	public void writeTopWordsWithProbability_Bar(){
		StringBuilder sBuilder = new StringBuilder();
		double[] phi_bar = estimatePhi_Bar();
		sBuilder.append("Background Topic: \n");
		for (int i = 0; i < topWordsOutputNumber; i++) {
			int max_index = FuncUtils.maxValueIndex(phi_bar);
			sBuilder.append(indexToWordMap.get(max_index) + " :" + phi_bar[max_index] + "\n");
			phi_bar[max_index] = 0;
		}
		try {
			FileUtil.writeFile(outputFileDirectory + "ILDA_backgroundtopic_word_" + K + ".txt", sBuilder.toString(),code);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	/**
	 * write theta for each document
	 */
	public void writeDocumentTopic(){
		double[][] theta = estimateTheta();
		StringBuilder sBuilder = new StringBuilder();
		for (int d = 0; d < theta.length; d++) {
			StringBuilder doc = new StringBuilder();
			for (int k = 0; k < theta[d].length; k++) {
				doc.append(theta[d][k] + "\t");
			}
			sBuilder.append(doc.toString().trim() + "\n");
		}
		try {
			FileUtil.writeFile(outputFileDirectory + "ILDA_doc_topic" + K + ".txt", sBuilder.toString(),code);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	/**
	 * write theta for each document
	 */
	public void writePerplexity(){
		StringBuilder sBuilder = new StringBuilder();
		double[][] theta = estimateTheta();
		double[][] phi = estimatePhi();
		double perplexity = Perplexity.lda_training_perplexity(docword, theta, phi);
		sBuilder.append(K + "\t Perplexity is: \n");
		sBuilder.append(perplexity);
		try {
			FileUtil.writeFile(outputFileDirectory + "ILDA_perplexity" + K + ".txt", sBuilder.toString(),code);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	public static void main(String args[]) throws Exception{
		ILDAGibbsSampling ilda = new ILDAGibbsSampling("data/shortdoc.txt", "gbk",20, 0.1,
				0.01,0.01, 0.1, 0.1, 1000, 100, "results/");
		ilda.MCMCSampling();
	}
}
