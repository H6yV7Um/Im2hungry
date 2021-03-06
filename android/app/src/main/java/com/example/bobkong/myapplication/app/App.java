package com.example.bobkong.myapplication.app;

import android.app.Application;
import android.os.Handler;

/**
 * Created by bobkong on 2018/6/8.
 */

public class App extends Application {

    /**
     * 启动照相Intent的RequestCode.自定义相机.
     */
    public static final int TAKE_PHOTO_CUSTOM = 100;
    /**
     * 启动照相Intent的RequestCode.系统相机.
     */
    public static final int TAKE_PHOTO_SYSTEM = 200;
    /**
     * 主线程Handler.
     */
    public static Handler mHandler;
    public static App sApp;

    @Override
    public void onCreate() {
        super.onCreate();
        sApp = this;
        mHandler = new Handler();
    }
}

