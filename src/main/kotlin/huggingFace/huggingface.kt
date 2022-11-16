import ai.djl.ModelException
import ai.djl.modality.nlp.qa.QAInput
import ai.djl.repository.zoo.Criteria
import ai.djl.training.util.ProgressBar
import ai.djl.translate.TranslateException
import java.io.IOException
import java.nio.file.Paths


object HuggingFaceQaInference {
    @Throws(IOException::class, TranslateException::class, ModelException::class)
    @JvmStatic
    fun main(args: Array<String>) {
        val question = "When did BBC Japan start broadcasting?"
        val paragraph = ("BBC Japan was a general entertainment Channel. "
                + "Which operated between December 2004 and April 2006. "
                + "It ceased operations after its Japanese distributor folded.")
        val input = QAInput(question, paragraph)
        val answer = qa_predict(input)
        println("The answer is: \n$answer")
    }

    @Throws(IOException::class, TranslateException::class, ModelException::class)
    fun qa_predict(input: QAInput?): String? {
        val translator = BertTranslator()
        val criteria = Criteria.builder()
                .setTypes(QAInput::class.java, String::class.java)
                .optModelPath(Paths.get("src/main/resources/trace_cased_bertqa/"))
                .optTranslator(translator)
                .optProgress(ProgressBar()).build()
        val model = criteria.loadModel()
        val predictor = model.newPredictor(translator)
        return predictor.predict(input)
    }
}