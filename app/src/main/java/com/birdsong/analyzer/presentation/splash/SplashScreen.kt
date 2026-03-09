package com.birdsong.analyzer.presentation.splash

import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.birdsong.analyzer.R

@Composable
fun SplashScreen(
    state: SplashViewModel.UiState,
    modifier: Modifier = Modifier,
) {
    val nameAlpha by animateFloatAsState(
        targetValue = if (state.phase >= 1) 1f else 0f,
        animationSpec = tween(durationMillis = 400),
        label = "nameAlpha",
    )
    val progressAlpha by animateFloatAsState(
        targetValue = if (state.phase >= 2) 1f else 0f,
        animationSpec = tween(durationMillis = 400),
        label = "progressAlpha",
    )
    val progressFraction by animateFloatAsState(
        targetValue = state.progress,
        animationSpec = tween(durationMillis = 150, easing = LinearEasing),
        label = "progressFraction",
    )

    Box(
        modifier = modifier
            .fillMaxSize()
            .background(SplashColors.Bg),
        contentAlignment = Alignment.Center,
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Image(
                painter = painterResource(R.drawable.ic_avalga_logo),
                contentDescription = null,
                modifier = Modifier.size(110.dp),
            )

            Spacer(modifier = Modifier.height(20.dp))

            Column(
                modifier = Modifier.alpha(nameAlpha),
                horizontalAlignment = Alignment.CenterHorizontally,
            ) {
                Text(
                    text = "AVALGA",
                    color = SplashColors.TextPrimary,
                    fontSize = 34.sp,
                    fontWeight = FontWeight.ExtraBold,
                    letterSpacing = 3.sp,
                )
                Text(
                    text = "BIRD SOUND ID",
                    color = SplashColors.TextSecondary,
                    fontSize = 11.sp,
                    letterSpacing = 4.sp,
                    modifier = Modifier.padding(top = 4.dp),
                )
            }

            Spacer(modifier = Modifier.height(40.dp))

            Column(
                modifier = Modifier
                    .alpha(progressAlpha)
                    .width(160.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
            ) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(2.dp)
                        .background(SplashColors.ProgressTrack, shape = RoundedCornerShape(2.dp)),
                ) {
                    Box(
                        modifier = Modifier
                            .fillMaxHeight()
                            .fillMaxWidth(fraction = progressFraction.coerceIn(0f, 1f))
                            .background(SplashColors.Accent, shape = RoundedCornerShape(2.dp)),
                    )
                }

                Spacer(modifier = Modifier.height(8.dp))

                Text(
                    text = stringResource(R.string.splash_loading),
                    color = SplashColors.TextMuted,
                    fontSize = 10.sp,
                    letterSpacing = 0.5.sp,
                )
            }
        }
    }
}

@Preview(showBackground = true, showSystemUi = true)
@Composable
private fun SplashPhase0Preview() {
    SplashScreen(state = SplashViewModel.UiState(phase = 0, progress = 0f))
}

@Preview(showBackground = true, showSystemUi = true)
@Composable
private fun SplashPhase2Preview() {
    SplashScreen(state = SplashViewModel.UiState(phase = 2, progress = 0.6f))
}
