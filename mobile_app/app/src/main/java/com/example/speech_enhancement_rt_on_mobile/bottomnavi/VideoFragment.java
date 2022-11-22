package com.example.speech_enhancement_rt_on_mobile.bottomnavi;

import android.os.Bundle;

import androidx.fragment.app.Fragment;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.VideoView;

import com.example.speech_enhancement_rt_on_mobile.R;

public class VideoFragment extends Fragment implements View.OnClickListener {

    private View v;
    VideoView videoView;
    ImageView bg_video;

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        v = inflater.inflate(R.layout.fragment_video, container, false);

        videoView = v.findViewById(R.id.videoView);
        videoView.setVisibility(View.INVISIBLE);
        bg_video = v.findViewById(R.id.bg_video);

        return v;
    }

    @Override
    public void onClick(View v) {
        switch (v.getId()) {
            /** video shoot */
            case R.id.btn_video_record:

                break;

            /** loading video */
            case R.id.btn_video_load:
                bg_video.setVisibility(View.INVISIBLE);
                videoView.setVisibility(View.VISIBLE);

                break;

            /** speech enhancement of video */
            case R.id.btn_video_se:


                break;

            /** play video */
            case R.id.btn_video_play:

                break;


        }
    }
}