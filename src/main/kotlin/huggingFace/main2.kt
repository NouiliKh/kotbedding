import ai.djl.modality.nlp.DefaultVocabulary
import ai.djl.modality.nlp.Vocabulary
import ai.djl.modality.nlp.bert.BertTokenizer
import ai.djl.modality.nlp.qa.QAInput
import ai.djl.ndarray.NDList
import ai.djl.translate.Batchifier
import ai.djl.translate.Translator
import ai.djl.translate.TranslatorContext
import java.io.IOException
import java.nio.file.Paths
import java.util.*

class BertTranslator : Translator<QAInput?, String?> {
    private var tokens: List<String>? = null
    private var vocabulary: Vocabulary? = null
    private var tokenizer: BertTokenizer? = null

    @Throws(IOException::class)
    override fun prepare(ctx: TranslatorContext) {
        val path = Paths.get("src/main/resources/bert-base-cased-vocab.txt")
        vocabulary = DefaultVocabulary.builder()
                .optMinFrequency(1)
                .addFromTextFile(path)
                .optUnknownToken("[UNK]")
                .build()
        tokenizer = BertTokenizer()
    }

    @Throws(IOException::class)
    override fun processInput(ctx: TranslatorContext?, p1: QAInput?): NDList? {
        val token = tokenizer!!.encode(
                p1?.question?.lowercase(Locale.getDefault()),
                p1?.paragraph?.lowercase(Locale.getDefault()))

        // get the encoded tokens that would be used in processOutput
        tokens = token.tokens
        val manager = ctx?.ndManager
        // map the tokens(String) to indices(long)
        val indices = (tokens as MutableList<String>?)?.stream()?.mapToLong { s: String? -> vocabulary!!.getIndex(s) }?.toArray()
        val attentionMask = token.attentionMask.stream().mapToLong { i: Long? -> i!! }.toArray()
        val tokenType = token.tokenTypes.stream().mapToLong { i: Long? -> i!! }.toArray()
        val indicesArray = manager?.create(indices)
        val attentionMaskArray = manager?.create(attentionMask)
        val tokenTypeArray = manager?.create(tokenType)
        // The order matters
        return NDList(indicesArray, attentionMaskArray, tokenTypeArray)
    }

    override fun processOutput(ctx: TranslatorContext, list: NDList): String {
        val startLogits = list[0]
        val endLogits = list[1]
        val startIdx = startLogits.argMax().getLong().toInt()
        val endIdx = endLogits.argMax().getLong().toInt()
        return tokenizer!!.tokenToString(tokens!!.subList(startIdx, endIdx + 1))
    }

    override fun getBatchifier(): Batchifier {
        return Batchifier.STACK
    }
}