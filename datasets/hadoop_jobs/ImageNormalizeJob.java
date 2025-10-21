import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
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
            URI[] cacheFiles = context.getCacheFiles();
            if (cacheFiles != null) {
                for (URI uri : cacheFiles) {
                    Path path = new Path(uri.getPath());
                    if (path.getName().equals("list_attr_celeba.csv")) {
                        try (BufferedReader br = new BufferedReader(new FileReader(path.getName()))) {
                            String line = br.readLine();
                            while ((line = br.readLine()) != null) {
                                String[] parts = line.split(",");
                                if (parts.length > 21) {
                                    String fileName = parts[0].trim();
                                    int male = Integer.parseInt(parts[21].trim());
                                    int blackHair = Integer.parseInt(parts[9].trim());
                                    if (male == -1 && blackHair == 1)
                                        validImages.add(fileName);
                                }
                            }
                        } catch (Exception e) {
                            System.err.println("Lỗi đọc CSV: " + e.getMessage());
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
            if (!validImages.contains(fileName)) return;

            try {
                ByteArrayInputStream bis = new ByteArrayInputStream(value.getBytes(), 0, value.getLength());
                BufferedImage img = ImageIO.read(bis);
                if (img == null) return;

                int width = 224, height = 224;
                Image tmp = img.getScaledInstance(width, height, Image.SCALE_SMOOTH);
                BufferedImage resized = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
                Graphics2D g2d = resized.createGraphics();
                g2d.drawImage(tmp, 0, 0, null);
                g2d.dispose();

                ByteArrayOutputStream bos = new ByteArrayOutputStream();
                ImageIO.write(resized, "jpg", bos);
                byte[] newBytes = bos.toByteArray();
                bos.close();

                context.write(new Text("female_blackhair/" + fileName), new BytesWritable(newBytes));

            } catch (Exception e) {
                System.err.println("Error processing file: " + fileName + " - " + e.getMessage());
            }
        }
    }

    public static class ImageReducer extends Reducer<Text, BytesWritable, Text, BytesWritable> {
        @Override
        protected void reduce(Text key, Iterable<BytesWritable> values, Context context)
                throws IOException, InterruptedException {

            int index = 0;
            for (BytesWritable val : values) {
                byte[] imageBytes = Arrays.copyOf(val.getBytes(), val.getLength());
                String outputName = key.toString();
                if (index > 0 && outputName.toLowerCase().endsWith(".jpg"))
                    outputName = outputName.replace(".jpg", "_" + index + ".jpg");

                context.write(new Text(outputName), new BytesWritable(imageBytes));
                index++;
            }
        }
    }

    // ================== MAIN ==================
    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: ImageNormalizeJob <input path> <output base path>");
            System.exit(-1);
        }

        String input = args[0];
        String baseOutput = args[1];

        int[][] configs = {
                {1, 1},
                {3, 1},
                {3, 3}
        };

        System.out.printf("%-20s %-20s %-20s %-20s %-20s\n",
                "Thành phần", "Thời gian bắt đầu(ms)", "Thời gian kết thúc(ms)", "Thời gian thực thi(ms)", "Ghi chú");

        for (int i = 0; i < configs.length; i++) {
            int mappers = configs[i][0];
            int reducers = configs[i][1];
            String jobName = String.format("Job_%dM_%dR", mappers, reducers);

            Configuration conf = new Configuration();
            Job job = Job.getInstance(conf, jobName);

            job.setJarByClass(ImageNormalizeJob.class);
            job.setMapperClass(ImageMapper.class);
            job.setReducerClass(ImageReducer.class);

            job.setInputFormatClass(CombineWholeFileInputFormat.class);
            job.setOutputFormatClass(WholeFileOutputFormat.class);

            job.setOutputKeyClass(Text.class);
            job.setOutputValueClass(BytesWritable.class);

            CombineWholeFileInputFormat.addInputPath(job, new Path(input));
            FileOutputFormat.setOutputPath(job, new Path(baseOutput + "_" + jobName));

            job.addCacheFile(new URI("/data_input/list_attr_celeba.csv#list_attr_celeba.csv"));
            job.setNumReduceTasks(reducers);

            long start = System.currentTimeMillis();
            boolean success = job.waitForCompletion(true);
            long end = System.currentTimeMillis();
            long duration = end - start;

            System.out.printf("%-20s %-20d %-20d %-20d %-20s\n",
                    jobName, start, end, duration,
                    success ? "Hoàn thành" : "Thất bại");
        }
    }
}
