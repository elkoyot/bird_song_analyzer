package com.birdsong.analyzer.ml

import kotlin.math.max

data class TimelineSegment(
    val scientificName: String,
    val commonName: String,
    val startTimeSec: Float,
    val endTimeSec: Float,
    val v24Confidence: Float?,
    val v30Confidence: Float?,
)

object TimelineBuilder {

    private data class RawInterval(
        val startTimeSec: Float,
        val endTimeSec: Float,
        val confidence: Float,
    )

    fun build(
        v24Result: BirdDetectionPipeline.FileAnalysisResult?,
        v30Result: BirdDetectionPipeline.FileAnalysisResult?,
        mergeGapSec: Float = 3.0f,
    ): List<TimelineSegment> {
        val v24Intervals = v24Result?.let { buildSpeciesIntervals(it) } ?: emptyMap()
        val v30Intervals = v30Result?.let { buildSpeciesIntervals(it) } ?: emptyMap()

        // Collect common names from both models
        val commonNames = HashMap<String, String>()
        v24Result?.chunkRecords?.forEach { r ->
            r.detections.forEach { d -> commonNames.putIfAbsent(d.scientificName, d.commonName) }
        }
        v30Result?.chunkRecords?.forEach { r ->
            r.detections.forEach { d -> commonNames.putIfAbsent(d.scientificName, d.commonName) }
        }

        val allSpecies = v24Intervals.keys + v30Intervals.keys
        val segments = mutableListOf<TimelineSegment>()

        for (species in allSpecies) {
            val v24Raw = v24Intervals[species] ?: emptyList()
            val v30Raw = v30Intervals[species] ?: emptyList()

            val v24Merged = mergeIntervals(v24Raw, mergeGapSec)
            val v30Merged = mergeIntervals(v30Raw, mergeGapSec)

            // Union time ranges from both models
            val allForUnion = mutableListOf<RawInterval>()
            for (iv in v24Merged) allForUnion.add(iv.copy(confidence = 0f))
            for (iv in v30Merged) allForUnion.add(iv.copy(confidence = 0f))
            val unionIntervals = mergeIntervals(allForUnion, mergeGapSec)

            for (union in unionIntervals) {
                val v24Conf = maxConfidenceOverlapping(v24Merged, union.startTimeSec, union.endTimeSec)
                val v30Conf = maxConfidenceOverlapping(v30Merged, union.startTimeSec, union.endTimeSec)

                segments.add(
                    TimelineSegment(
                        scientificName = species,
                        commonName = commonNames[species] ?: species,
                        startTimeSec = union.startTimeSec,
                        endTimeSec = union.endTimeSec,
                        v24Confidence = if (v24Result != null) v24Conf else null,
                        v30Confidence = if (v30Result != null) v30Conf else null,
                    )
                )
            }
        }

        return segments.sortedBy { it.startTimeSec }
    }

    private fun buildSpeciesIntervals(
        result: BirdDetectionPipeline.FileAnalysisResult,
    ): Map<String, List<RawInterval>> {
        val map = HashMap<String, MutableList<RawInterval>>()
        for (record in result.chunkRecords) {
            val end = record.startTimeSec + result.chunkDurationSec
            for (det in record.detections) {
                map.getOrPut(det.scientificName) { mutableListOf() }
                    .add(RawInterval(record.startTimeSec, end, det.confidence))
            }
        }
        return map
    }

    private fun mergeIntervals(
        intervals: List<RawInterval>,
        mergeGapSec: Float,
    ): List<RawInterval> {
        if (intervals.isEmpty()) return emptyList()
        val sorted = intervals.sortedBy { it.startTimeSec }
        val result = mutableListOf<RawInterval>()

        var current = sorted[0]
        for (i in 1 until sorted.size) {
            val next = sorted[i]
            if (next.startTimeSec <= current.endTimeSec + mergeGapSec) {
                current = RawInterval(
                    startTimeSec = current.startTimeSec,
                    endTimeSec = max(current.endTimeSec, next.endTimeSec),
                    confidence = max(current.confidence, next.confidence),
                )
            } else {
                result.add(current)
                current = next
            }
        }
        result.add(current)
        return result
    }

    private fun maxConfidenceOverlapping(
        merged: List<RawInterval>,
        start: Float,
        end: Float,
    ): Float? {
        var maxConf: Float? = null
        for (iv in merged) {
            if (iv.startTimeSec < end && iv.endTimeSec > start) {
                maxConf = max(maxConf ?: 0f, iv.confidence)
            }
        }
        return maxConf
    }
}
