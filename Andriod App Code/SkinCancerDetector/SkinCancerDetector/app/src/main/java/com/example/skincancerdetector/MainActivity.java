package com.example.skincancerdetector;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
 import androidx.annotation.Nullable;
 import androidx.annotation.RequiresApi;
 import androidx.appcompat.app.AppCompatActivity;
 import android.Manifest;
 import android.content.Intent;
 import android.content.pm.PackageManager;
 import android.graphics.Bitmap;
 import android.media.ThumbnailUtils;
 import com.example.skincancerdetector.ml.Model;
 import org.tensorflow.lite.DataType;
 import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
 import java.io.IOException;
 import java.nio.ByteBuffer;
 import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {

    TextView result, confidence;
    ImageView imageView;
    Button picture;
    int imageSize = 224;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        result = findViewById(R.id.result);
        confidence = findViewById(R.id.confidence);
        imageView = findViewById(R.id.imageView);
        picture = findViewById(R.id.button);

        picture.setOnClickListener(new View.OnClickListener() {
            @RequiresApi(api = Build.VERSION_CODES.M)
            @Override
            public void onClick(View view) {
                // Launch camera if we have permission
                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, 1);
                } else {
                    //Request camera permission if we don't have it.
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                }
            }
        });
    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if (requestCode == 1 && resultCode == RESULT_OK) {


            Bitmap image = (Bitmap) data.getExtras().get("data");

            int dimention = Math.min(image.getWidth(), image.getHeight());

            image = ThumbnailUtils.extractThumbnail(image, dimention, dimention);

            image = Bitmap.createScaledBitmap(image, 244, 244, false);

            classifyimage(image);


        }
        super.onActivityResult(requestCode, resultCode, data);
    }


    private void classifyimage(Bitmap image) {

        try {
            Model model = Model.newInstance(getApplicationContext());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, imageSize, imageSize, 3}, DataType.FLOAT32);

            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());
            int[] intvalues = new int[244 * 244];
            image.getPixels(intvalues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());

            int pixel = 0;
            for (int i = 0; i < imageSize; i++) {
                for (int j = 0; j < imageSize; j++) {
                    int val = intvalues[pixel++];// RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 255.f));

                }

            }

            inputFeature0.loadBuffer(byteBuffer);
            // Runs model inference and gets result.
            Model.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();


            float[] acc = outputFeature0.getFloatArray();
            int m = 0;
            float maxacc = 0;
            for (int i = 0; i < acc.length; i++) {
                if (acc[i] > maxacc) {
                    maxacc = acc[i];
                    m = i;

                };
            };

            String[] classes = {"High Risk: Melanoma", "low Risk: Nevus", "Median Risk: Squamous cell carcinoma", "low Risk: Benign"};
            result.setText(classes[m]);

            String s = "";
            for (int i = 0; i < classes.length; i++) {
                s+=String.format("%s  :%.1f%%\n", classes[i], acc[i] * 100);

            }
            confidence.setText(s);


            model.close();
        } catch (IOException e) {

        }


    }
}