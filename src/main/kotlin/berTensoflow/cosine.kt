
fun cosineSimilarity(vectorA: FloatArray, vectorB: FloatArray): Double {
    var dotProduct = 0.0
    var normA = 0.0
    var normB = 0.0
    for (i in vectorA.indices) {
        dotProduct += vectorA[i] * vectorB[i]
        normA += Math.pow(vectorA[i].toDouble(), 2.0)
        normB += Math.pow(vectorB[i].toDouble(), 2.0)
    }
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB))
}