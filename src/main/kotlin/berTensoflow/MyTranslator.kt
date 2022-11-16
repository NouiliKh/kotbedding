import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDArrays
import ai.djl.ndarray.NDList
import ai.djl.translate.NoBatchifyTranslator
import ai.djl.translate.TranslatorContext
import java.util.*
import java.util.stream.Collectors

val MyTranslator = object : NoBatchifyTranslator<Array<String>, Array<FloatArray>> {

    override fun processInput(p0: TranslatorContext?, p1: Array<String>?): NDList? {
        val manager = p0?.ndManager
        val inputsList = NDList(
                Arrays.stream(p1)
                        .map { data: String? -> manager?.create(data) }
                        .collect(Collectors.toList()))
        return NDList(NDArrays.stack(inputsList))
    }


    override fun processOutput(ctx: TranslatorContext, list: NDList): Array<FloatArray> {
        val result = NDList()
        val numOutputs = list.singletonOrThrow().shape[0]
        for (i in 0 until numOutputs) {
            result.add(list.singletonOrThrow()[i])
        }
        val a = result
        val c = result.stream()
        val result1 = result.map { obj: NDArray? -> obj?.toFloatArray()?:FloatArray(0) }
        return result1.toTypedArray();

    }
}