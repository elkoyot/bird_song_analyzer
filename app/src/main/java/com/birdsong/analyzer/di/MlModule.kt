package com.birdsong.analyzer.di

import android.content.Context
import com.birdsong.analyzer.ml.AudioChunkProcessor
import com.birdsong.analyzer.ml.BirdClassifier
import com.birdsong.analyzer.ml.PreprocessingMode
import com.birdsong.analyzer.ml.BirdNetV24Classifier
import com.birdsong.analyzer.ml.CountryConfig
import com.birdsong.analyzer.ml.CountryConfigLoader
import com.birdsong.analyzer.ml.FamilyTaxonomy
import com.birdsong.analyzer.ml.LabelParser
import com.birdsong.analyzer.ml.MetaProfileBuilder
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import javax.inject.Named
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object MlModule {

    @Provides
    @Singleton
    fun provideAudioChunkProcessor(): AudioChunkProcessor = AudioChunkProcessor(mode = PreprocessingMode.PASSTHROUGH)

    @Provides
    @Singleton
    @Named("birdnetAudioModel")
    fun provideAudioModel(@ApplicationContext context: Context): MappedByteBuffer =
        loadModel(context, BirdNetV24Classifier.AUDIO_MODEL_PATH)

    @Provides
    @Singleton
    @Named("birdnetMetaModel")
    fun provideMetaModel(@ApplicationContext context: Context): MappedByteBuffer =
        loadModel(context, BirdNetV24Classifier.META_MODEL_PATH)

    @Provides
    @Singleton
    @Named("birdnetLabels")
    fun provideLabels(@ApplicationContext context: Context): List<Pair<String, String>> {
        val path = "${BirdNetV24Classifier.ASSET_BASE}/labels/ru.txt"
        return context.assets.open(path).use { LabelParser.load(it) }
    }

    @Provides
    @Singleton
    fun provideCountries(@ApplicationContext context: Context): List<CountryConfig> =
        CountryConfigLoader.load(context)

    @Provides
    @Singleton
    fun provideBirdClassifier(
        @Named("birdnetAudioModel") audioModel: MappedByteBuffer,
        @Named("birdnetMetaModel") metaModel: MappedByteBuffer,
        @Named("birdnetLabels") labels: List<Pair<String, String>>,
    ): BirdClassifier = BirdNetV24Classifier(audioModel, metaModel, labels)

    @Provides
    @Singleton
    fun provideFamilyTaxonomy(@ApplicationContext context: Context): FamilyTaxonomy =
        FamilyTaxonomy.loadFromAssets(context)

    @Provides
    @Singleton
    fun provideMetaProfileBuilder(
        @Named("birdnetMetaModel") metaModel: MappedByteBuffer,
        @Named("birdnetLabels") labels: List<Pair<String, String>>,
    ): MetaProfileBuilder = MetaProfileBuilder(metaModel.asReadOnlyBuffer(), labels.size)

    private fun loadModel(context: Context, assetPath: String): MappedByteBuffer {
        return context.assets.openFd(assetPath).use { fd ->
            FileInputStream(fd.fileDescriptor).use { fis ->
                fis.channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
            }
        }
    }
}
