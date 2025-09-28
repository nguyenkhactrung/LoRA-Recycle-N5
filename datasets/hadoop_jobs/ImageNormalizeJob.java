import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.net.URI;
import java.util.*;

public class ImageNormalizeJob {

    public static class ImageMapper extends Mapper<Text, BytesWritable, Text, BytesWritable> {

        private Set<String> validImages = new HashSet<>();

        @Override
        protected void setup(Context context) throws IOException {
            // Lấy metadata từ DistributedCache
            URI[] cacheFiles = context.getCacheFiles();
            if (cacheFiles != null) {
                for (URI uri : cacheFiles) {
                    Path path = new Path(uri.getPath());
                    if (path.getName().equals("list_attr_celeba.csv")) {
                        try (BufferedReader br = new BufferedReader(new FileReader(path.getName()))) {
                            String line = br.readLine(); // bỏ header
                            while ((line = br.readLine()) != null) {
                                String[] parts = line.split(",");
                                if (parts.length > 3) {
                                    String fileName = parts[0].trim();
                                    int male = Integer.parseInt(parts[21].trim()); // Male
                                    int blackHair = Integer.parseInt(parts[9].trim()); // Black_Hair

                                    // Giữ ảnh nếu là nữ (male == -1) và tóc đen (blackHair == 1)
                                    if (male == -1 && blackHair == 1) {
                                        validImages.add(fileName);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        @Override
        protected void map(Text key, BytesWritable value, Context context)
                throws IOException, InterruptedException {

            String filePath = key.toString();
            String fileName = filePath.substring(filePath.lastIndexOf("/") + 1);

            if (validImages.contains(fileName)) {
                try {
                    // ---- 1. Chuyển bytes -> BufferedImage ----
                    ByteArrayInputStream bis = new ByteArrayInputStream(value.getBytes(), 0, value.getLength());
                    BufferedImage img = ImageIO.read(bis);
                    if (img == null)
                        return;

                    // ---- 2. Resize về 224x224 ----
                    int width = 224, height = 224;
                    Image tmp = img.getScaledInstance(width, height, Image.SCALE_SMOOTH);
                    BufferedImage resized = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
                    Graphics2D g2d = resized.createGraphics();
                    g2d.drawImage(tmp, 0, 0, null);
                    g2d.dispose();

                    // ---- 3. Encode lại thành byte[] ----
                    ByteArrayOutputStream bos = new ByteArrayOutputStream();
                    ImageIO.write(resized, "jpg", bos);
                    bos.flush();
                    byte[] newBytes = bos.toByteArray();
                    bos.close();

                    // ---- 4. Ghi ra HDFS ----
                    context.write(new Text(fileName), new BytesWritable(newBytes));

                } catch (Exception e) {
                    System.err.println("Error processing file: " + fileName + " - " + e.getMessage());
                }
            }
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: ImageNormalizeJob <input path> <output path>");
            System.exit(-1);
        }

        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "Image Resize with Filter");

        job.setJarByClass(ImageNormalizeJob.class);
        job.setMapperClass(ImageMapper.class);
        job.setNumReduceTasks(0);

        job.setInputFormatClass(CombineWholeFileInputFormat.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(BytesWritable.class);

        CombineWholeFileInputFormat.addInputPath(job, new Path(args[0]));
        job.setOutputFormatClass(WholeFileOutputFormat.class);
        WholeFileOutputFormat.setOutputPath(job, new Path(args[1]));

        // Thêm file metadata (CSV) vào DistributedCache
        job.addCacheFile(new URI("/data_input/list_attr_celeba.csv#list_attr_celeba.csv"));

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
