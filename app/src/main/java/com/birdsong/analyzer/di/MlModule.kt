package com.birdsong.analyzer.di

import android.content.Context
import com.birdsong.analyzer.ml.AudioChunkProcessor
import com.birdsong.analyzer.ml.BirdClassifier
import com.birdsong.analyzer.ml.BirdNetV24Classifier
import com.birdsong.analyzer.ml.LabelParser
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object MlModule {

    @Provides
    @Singleton
    fun provideAudioChunkProcessor(): AudioChunkProcessor = AudioChunkProcessor()

    @Provides
    @Singleton
    fun provideBirdClassifier(@ApplicationContext context: Context): BirdClassifier {
        val audioModel = loadModel(context, BirdNetV24Classifier.AUDIO_MODEL_PATH)
        val metaModel = loadModel(context, BirdNetV24Classifier.META_MODEL_PATH)

        val labelsPath = "${BirdNetV24Classifier.ASSET_BASE}/labels/ru.txt"
        val labels = context.assets.open(labelsPath).use { LabelParser.load(it) }

        return BirdNetV24Classifier(audioModel, metaModel, labels)
    }

    private fun loadModel(context: Context, assetPath: String): MappedByteBuffer {
        return context.assets.openFd(assetPath).use { fd ->
            FileInputStream(fd.fileDescriptor).use { fis ->
                fis.channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
            }
        }
    }
}
