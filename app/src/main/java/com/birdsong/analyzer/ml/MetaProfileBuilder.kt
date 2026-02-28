package com.birdsong.analyzer.ml

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer

/**
 * Строит [MetaProfile] для указанного bbox: разворачивает сетку точек (gridStepDeg),
 * прогоняет мета-модель для каждой точки × каждой недели и берёт максимум per species.
 *
 * Запускается один раз в фоне при старте LiveDetectionViewModel.
 * Для Беларуси (~35 точек × 52 недели ≈ 1820 inference) занимает 2–3 сек.
 */
class MetaProfileBuilder(
    metaModel: ByteBuffer,
    private val speciesCount: Int,
) {
    private val interpreter = Interpreter(metaModel, Interpreter.Options().apply { numThreads = 2 })

    suspend fun build(
        bbox: BoundingBox,
        bufferDeg: Float = 2.5f,
        weekRange: IntRange = 1..52,
        gridStepDeg: Float = 3.0f,
    ): MetaProfile = withContext(Dispatchers.Default) {
        val minLat = (bbox.minLat - bufferDeg).coerceAtLeast(-90f)
        val maxLat = (bbox.maxLat + bufferDeg).coerceAtMost(90f)
        val minLon = (bbox.minLon - bufferDeg).coerceAtLeast(-180f)
        val maxLon = (bbox.maxLon + bufferDeg).coerceAtMost(180f)

        val latSteps = ((maxLat - minLat) / gridStepDeg).toInt() + 1
        val lonSteps = ((maxLon - minLon) / gridStepDeg).toInt() + 1
        val totalPoints = latSteps * lonSteps
        val totalInferences = totalPoints * (weekRange.last - weekRange.first + 1)

        Log.d(TAG, "Building MetaProfile: bbox=[$minLat,$maxLat]×[$minLon,$maxLon] " +
            "${latSteps}×${lonSteps}=$totalPoints pts × ${weekRange.count()} weeks = $totalInferences inf")

        val maxScores = FloatArray(speciesCount)
        val metaInput = arrayOf(FloatArray(3))
        val metaOutput = Array(1) { FloatArray(speciesCount) }

        for (latIdx in 0 until latSteps) {
            val lat = minLat + latIdx * gridStepDeg
            for (lonIdx in 0 until lonSteps) {
                val lon = minLon + lonIdx * gridStepDeg
                metaInput[0][0] = lat
                metaInput[0][1] = lon
                for (week in weekRange) {
                    metaInput[0][2] = week.toFloat()
                    interpreter.run(metaInput, metaOutput)
                    val weekScores = metaOutput[0]
                    for (i in maxScores.indices) {
                        if (weekScores[i] > maxScores[i]) maxScores[i] = weekScores[i]
                    }
                }
            }
        }

        Log.d(TAG, "MetaProfile built: $totalInferences inferences done")
        MetaProfile(maxScores)
    }

    fun close() {
        interpreter.close()
    }

    companion object {
        private const val TAG = "MetaProfileBuilder"
    }
}
