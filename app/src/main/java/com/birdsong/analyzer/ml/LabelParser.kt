package com.birdsong.analyzer.ml

import java.io.InputStream

object LabelParser {

    /**
     * Loads labels from an input stream.
     * Each line format: "ScientificName_CommonName"
     *
     * @param inputStream stream with label lines
     * @return list of (scientificName, commonName) pairs, index matches model output
     */
    fun load(inputStream: InputStream): List<Pair<String, String>> =
        inputStream.bufferedReader().use { reader ->
            reader.lineSequence()
                .filter { it.isNotBlank() }
                .map { line ->
                    val sep = line.indexOf('_')
                    if (sep == -1) {
                        line to line
                    } else {
                        line.substring(0, sep) to line.substring(sep + 1)
                    }
                }
                .toList()
        }
}
