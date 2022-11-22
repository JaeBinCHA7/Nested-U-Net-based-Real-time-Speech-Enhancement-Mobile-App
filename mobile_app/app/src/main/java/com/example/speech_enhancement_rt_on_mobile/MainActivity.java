package com.example.speech_enhancement_rt_on_mobile;

import androidx.annotation.NonNull;
import androidx.appcompat.app.ActionBar;
import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.view.MenuItem;

import com.example.speech_enhancement_rt_on_mobile.bottomnavi.AudioFragment;
import com.example.speech_enhancement_rt_on_mobile.bottomnavi.VideoFragment;
import com.google.android.material.bottomnavigation.BottomNavigationView;

public class MainActivity extends AppCompatActivity {
    BottomNavigationView bottomNavigationView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        ActionBar actionBar = getSupportActionBar();
        actionBar.hide();

        bottomNavigationView = findViewById(R.id.bottomnavi);
        getSupportFragmentManager().beginTransaction().add(R.id.fl_main, new AudioFragment()).commit();
        bottomNavigationView.setOnNavigationItemSelectedListener(new BottomNavigationView.OnNavigationItemSelectedListener() {
            @Override
            public boolean onNavigationItemSelected(@NonNull MenuItem menuItem) {
                switch (menuItem.getItemId()) {
                    case R.id.bottom_audio:
                        getSupportFragmentManager().beginTransaction().replace(R.id.fl_main, new AudioFragment()).commit();
                        break;
                    case R.id.bottom_video:
                        getSupportFragmentManager().beginTransaction().replace(R.id.fl_main, new VideoFragment()).commit();
                        break;
                }
                return true;
            }
        });

    }
}