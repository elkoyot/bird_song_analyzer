package com.birdsong.analyzer.ml

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class LabelParserTest {

    @Test
    fun `parses labels with underscore separator`() {
        val input = """
            Parus major_Great Tit
            Turdus merula_Common Blackbird
        """.trimIndent()

        val labels = LabelParser.load(input.byteInputStream())

        assertEquals(2, labels.size)
        assertEquals("Parus major" to "Great Tit", labels[0])
        assertEquals("Turdus merula" to "Common Blackbird", labels[1])
    }

    @Test
    fun `handles label without underscore`() {
        val input = "UnknownSpecies\n"

        val labels = LabelParser.load(input.byteInputStream())

        assertEquals(1, labels.size)
        assertEquals("UnknownSpecies" to "UnknownSpecies", labels[0])
    }

    @Test
    fun `skips blank lines`() {
        val input = """
            Parus major_Great Tit

            Corvus corax_Common Raven
        """.trimIndent()

        val labels = LabelParser.load(input.byteInputStream())

        assertEquals(2, labels.size)
    }

    @Test
    fun `handles label with multiple underscores`() {
        val input = "Parus_major_Great Tit\n"

        val labels = LabelParser.load(input.byteInputStream())

        // Only splits on first underscore
        assertEquals("Parus" to "major_Great Tit", labels[0])
    }

    @Test
    fun `returns empty list for empty input`() {
        val labels = LabelParser.load("".byteInputStream())

        assertEquals(emptyList<Pair<String, String>>(), labels)
    }
}
