package com.birdsong.analyzer.ml

/**
 * Предвычисленный профиль мета-модели для конкретного региона.
 *
 * Хранит максимальные скоры мета-модели по сетке точек страны + весь год.
 * Строится один раз при старте сессии через [MetaProfileBuilder].
 *
 * Tiered alpha: разный вес в зависимости от того, насколько вид "ожидаем" в регионе:
 * - m >= TIER_COMMON    → baseAlpha (обычный региональный вид)
 * - m >= TIER_IRRUPTIVE → 0.50 (инвазионный / пограничный)
 * - m >= TIER_VAGRANT   → 0.25 (очень редкий залётный)
 * - else                → ALPHA_OUTLIER (континентальный выброс)
 */
class MetaProfile(val maxScores: FloatArray) {

    fun apply(scores: FloatArray, baseAlpha: Float) {
        require(scores.size == maxScores.size) {
            "scores.size=${scores.size} != maxScores.size=${maxScores.size}"
        }
        for (i in scores.indices) {
            val m = maxScores[i]
            val ea = when {
                m >= TIER_COMMON     -> baseAlpha
                m >= TIER_IRRUPTIVE  -> 0.50f
                m >= TIER_VAGRANT    -> 0.25f
                else                 -> ALPHA_OUTLIER
            }
            scores[i] *= ea + (1f - ea) * m
        }
    }

    fun tierLabel(index: Int): String {
        val m = maxScores.getOrElse(index) { return "?" }
        return tierLabel(m)
    }

    companion object {
        const val TIER_COMMON     = 0.30f
        const val TIER_IRRUPTIVE  = 0.05f
        const val TIER_VAGRANT    = 0.01f
        const val ALPHA_OUTLIER   = 0.02f

        fun tierLabel(m: Float): String = when {
            m >= TIER_COMMON    -> "C"   // Common
            m >= TIER_IRRUPTIVE -> "I"   // Irruptive
            m >= TIER_VAGRANT   -> "V"   // Vagrant
            else                -> "O"   // Outlier
        }
    }
}
