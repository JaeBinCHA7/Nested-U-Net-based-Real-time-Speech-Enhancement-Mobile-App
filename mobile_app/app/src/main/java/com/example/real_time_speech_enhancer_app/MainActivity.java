package com.example.real_time_speech_enhancer_app;

import androidx.annotation.NonNull;
import androidx.appcompat.app.ActionBar;
import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.view.MenuItem;

import com.example.real_time_speech_enhancer_app.bottomnavi.BaselineFragment;
import com.example.real_time_speech_enhancer_app.bottomnavi.ProposedFragement;
import com.google.android.material.bottomnavigation.BottomNavigationView;

public class MainActivity extends AppCompatActivity {
    BottomNavigationView bottomNavigationView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        ActionBar actionBar = getSupportActionBar();
        actionBar.hide(); // remove action bar

        bottomNavigationView = findViewById(R.id.bottomnavi);
        getSupportFragmentManager().beginTransaction().add(R.id.fl_main, new ProposedFragement()).commit(); // Home

        bottomNavigationView.setOnNavigationItemSelectedListener(new BottomNavigationView.OnNavigationItemSelectedListener() { // Set items inside the bottom navigation view
            @Override
            public boolean onNavigationItemSelected(@NonNull MenuItem menuItem) {
                switch (menuItem.getItemId()) {
                    case R.id.proposed_model:
                        getSupportFragmentManager().beginTransaction().replace(R.id.fl_main, new ProposedFragement()).commit();
                        break;
                    case R.id.baseline:
                        getSupportFragmentManager().beginTransaction().replace(R.id.fl_main, new BaselineFragment()).commit();
                        break;
                }
                return true;
            }
        });

    }
}