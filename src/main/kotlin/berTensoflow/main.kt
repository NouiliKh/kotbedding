import MyTranslator
import ai.djl.Application
import ai.djl.MalformedModelException
import ai.djl.ModelException
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDArrays
import ai.djl.ndarray.NDList
import ai.djl.repository.zoo.Criteria
import ai.djl.repository.zoo.ModelNotFoundException
import ai.djl.training.util.ProgressBar
import ai.djl.translate.NoBatchifyTranslator
import ai.djl.translate.TranslateException
import ai.djl.translate.TranslatorContext
import org.slf4j.LoggerFactory
import java.io.IOException
import java.util.*
import java.util.stream.Collectors
import cosineSimilarity



object UniversalSentenceEncoder {
    val logger = LoggerFactory.getLogger(UniversalSentenceEncoder::class.java)
    @Throws(IOException::class, ModelException::class, TranslateException::class)
    @JvmStatic
    fun main(args: Array<String>) {
        val inputs: MutableList<String> = ArrayList()
        inputs.add("Marcel Hamböcker Kfz-Service Kfz-Service Hamböcker")
        inputs.add("Kfz-Service Hamböcker")
        val embeddings = predict(inputs)
        val a = cosineSimilarity(embeddings[0], embeddings[1])
    }

    @Throws(MalformedModelException::class, ModelNotFoundException::class, IOException::class, TranslateException::class)
    fun predict(inputs: List<String>): Array<FloatArray> {
        val modelUrl = "https://storage.googleapis.com/tfhub-modules/google/universal-sentence-encoder/4.tar.gz"
        val translate = MyTranslator
        val criteria = Criteria.builder()
                .optApplication(Application.NLP.TEXT_EMBEDDING)
                .setTypes(NDList::class.java, NDList::class.java)
                .optModelUrls(modelUrl)
                .optEngine("TensorFlow")
                .optProgress(ProgressBar())
                .build()

        val model = criteria.loadModel()
        val predictor = model.newPredictor(translate)
        val a = predictor.predict(inputs.toTypedArray())
        return  a
    }
}





