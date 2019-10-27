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
	public double alpha ;  //������ alpha0 = 1E-12
	public double beta; 
	public double beta_back;
	public double gamma0;  //for beta distribution 
	double gamma1; //for beta distribution 
	int[][] z;
	public int[][] ndk; // document-topic count
	public int[] ndsum; //document-topic sum
	public int[][] nkw; //topic-word count
	public int[] nksum; //topic-word sum (total number of words assigned to a topic)
	boolean c[][]; //������ѡ����
	public long[] n_cv; //2ά��  ������0��Ӧ���ܵ�������  �� �Ǳ�����1��Ӧ���ܵ�������
	public int[] nback_v; //1*V ������ĳ���ʵ�Ƶ��
	JDKRandomGenerator rand; //�����������
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
		//�ĵ�d������k���ɵĵ�����Ŀ
		ndk = new int[M][K];
		//ÿƪ�ĵ����������е�������
		ndsum = new int[M];
		//����k�е���v����Ŀ
		nkw = new int[K][V];
		//����k��Ӧ�ĵ�������
		nksum = new int[K];
		//ÿƪ�ĵ����ʶ�Ӧ������
		z = new int[M][];
		//ĳ���Ƿ�Ϊ������0-1��ʼ��
		c = new boolean[M][];
		//�����ʺͷǱ���������ͳ��
		n_cv = new long[2];
		//����ͳ�� ĳ�����ڱ�����
		nback_v = new int[V];
		// assign topics
		for (int d = 0; d < M; d++) {
			// words
			int Nd = docword[d].length;
			z[d] = new int[Nd];
			//ѭ��ÿһ������
			for (int n = 0; n < Nd; n++) {
				//�����ֵ����
				int topic = (int) (Math.random() * K);
				z[d][n] = topic;
			}
		}
		//assign label c
		for (int d = 0; d < M; d++) {
			c[d] = new boolean[docword[d].length];
			for (int n = 0; n < docword[d].length; n++) {
				if (Math.random() > 0.5) {
					c[d][n] = true;  //true��ʾ���Ǳ�����
					//����ͳ��
					updateCount(d, z[d][n], docword[d][n], +1);
				} else {   //����Ǳ�����
					c[d][n] = false;
					updateCountBackWord(docword[d][n], 1);
				}

			}
		}
	}
	public void MCMCSampling() {
		//��������
		for (int i = 0; i < this.iterations; i++) {
			System.out.println("iteration : " + i);
			gibbsOneIteration();  //ִ��gibbs����
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
	//gibbs����
	public void gibbsOneIteration() {
		//sample topic
		for (int d = 0; d < docword.length; d++) {
			//ѭ�����е���
			for (int n = 0; n < z[d].length; n++) {
				int topic = sampleFullConditional(d, n);
				//�ȳ�ȡ����
				z[d][n] = topic;
				if (c[d][n]) {  //��ʾ���Ǳ�����
					ndk[d][topic] += 1;
					//�ĵ�d
					ndsum[d] += 1;
					//����topic��Ӧ�ĵ���word������1
					nkw[topic][docword[d][n]] += 1;
					//����topic��Ӧ�ĵ���������1
					nksum[topic] += 1;
				}
			}
		}
		for (int d = 0; d < docword.length; d++) {
			//ѭ�����е���
			for (int n = 0; n < c[d].length; n++) {
				sample_label(d, n);
			}
		}
		//����ͳ����Ŀ
		cleanTempPrmts();
		for (int d = 0; d < docword.length; d++) {
			//ѭ�����е���
			for (int n = 0; n < c[d].length; n++) {
				if (c[d][n]) {
					//����ͳ��
					updateCount(d, z[d][n], docword[d][n], +1);
				}else {
					updateCountBackWord(docword[d][n], 1);
				}
			}
		}
	}
	//��Բ�Ϊ�����ʵĵ��ʳ�ȡ����
	int sampleFullConditional(int d, int n) {
		//��ȡԭ��Ӧ������
		int topic = z[d][n];
		if (c[d][n]) { //�����Ϊ������
			ndk[d][topic] += -1;
			//�ĵ�d
			ndsum[d] += -1;
			//����topic��Ӧ�ĵ���word������1
			nkw[topic][docword[d][n]] += -1;
			//����topic��Ӧ�ĵ���������1
			nksum[topic] += -1;
		}
		//����
		double[] p = new double[K];
		//ѭ��ÿ������
		for (int k = 0; k < K; k++) {
			p[k] = (ndk[d][k] + alpha) / (ndsum[d] + K * alpha) * (nkw[k][docword[d][n]] + beta)
					/ (nksum[k] + V * beta);
		}
		//���̶ĳ�ȡ������
		topic = sample(p);
		//��������
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
		if (binary == 0) {  //����Ǳ�����
			nback_v[docword[d][n]]--;
		} else {   //������Ǳ�����
			ndk[d][z[d][n]]--;
			//�ĵ�d
			ndsum[d]--;
			//����topic��Ӧ�ĵ���word������1
			nkw[z[d][n]][docword[d][n]]--;
			//����topic��Ӧ�ĵ���������1
			nksum[z[d][n]]--;
		}
		binarylabel = draw_label(d, n);
		c[d][n] = binarylabel;
	}
	public void cleanTempPrmts() {
		ndk = new int[M][K];
		ndsum = new int[M];
		//����k�е���v����Ŀ
		nkw = new int[K][V];
		//����k��Ӧ�ĵ�������
		nksum = new int[K];
		//ÿƪ�ĵ����ʶ�Ӧ������
		n_cv = new long[2];
		//����ͳ�� ĳ�����ڱ�����
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
	//���̶�
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
	//����ͳ��
	void updateCount(int d, int topic, int word, int flag) {
		//�ĵ� d�е�����topic��Ӧ�ĵ�����Ŀ��1
		ndk[d][topic] += flag;
		//�ĵ�d
		ndsum[d] += flag;
		//����topic��Ӧ�ĵ���word������1
		nkw[topic][word] += flag;
		//����topic��Ӧ�ĵ���������1
		nksum[topic] += flag;
		//���Ǳ����ʵĵ�����Ŀ
		n_cv[1] += flag; 
	}
	void updateCount_New(int d, int topic, int word, int flag) {
		//�ĵ� d�е�����topic��Ӧ�ĵ�����Ŀ��1
		ndk[d][topic] += flag;
		//�ĵ�d
		ndsum[d] += flag;
		//����topic��Ӧ�ĵ���word������1
		nkw[topic][word] += flag;
		//����topic��Ӧ�ĵ���������1
		nksum[topic] += flag;
		//���Ǳ����ʵĵ�����Ŀ
		n_cv[1] += flag; 
	}
	//����ͳ��  ������
	void updateCountBackWord(int word, int flag) {
		nback_v[word] += flag;
		n_cv[0] += flag;  //��������Ŀͳ��
	}
	//����Theta
	public double[][] estimateTheta() {
		double[][] theta = new double[docword.length][K];
		for (int d = 0; d < docword.length; d++) {
			for (int k = 0; k < K; k++) {
				theta[d][k] = (ndk[d][k] + + alpha) / (ndsum[d] + + K * alpha);
			}
		}
		return theta;
	}
	//����Phi
	public double[][] estimatePhi() {
		double[][] phi = new double[K][V];
		for (int k = 0; k < K; k++) {
			for (int w = 0; w < V; w++) {
				phi[k][w] = (nkw[k][w] + beta) / (nksum[k] + V * beta);
			}
		}
		return phi;
	}
	//����Phi
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
