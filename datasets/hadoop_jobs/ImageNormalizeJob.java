import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import javax.imageio.ImageIO;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;

public class ImageNormalizeJob extends Configured implements Tool {

    // Mapper
    public static class ImageMapper extends Mapper<Text, BytesWritable, Text, BytesWritable> {
        @Override
        protected void map(Text key, BytesWritable value, Context context)
                throws IOException, InterruptedException {

            byte[] imageBytes = value.getBytes();
            ByteArrayInputStream bis = new ByteArrayInputStream(imageBytes);
            BufferedImage img = ImageIO.read(bis);

            if (img != null) {
                // Resize 224x224
                Image scaled = img.getScaledInstance(224, 224, Image.SCALE_SMOOTH);
                BufferedImage resized = new BufferedImage(224, 224, BufferedImage.TYPE_INT_RGB);
                Graphics2D g2d = resized.createGraphics();
                g2d.drawImage(scaled, 0, 0, null);
                g2d.dispose();

                ByteArrayOutputStream bos = new ByteArrayOutputStream();
                ImageIO.write(resized, "jpg", bos);
                byte[] processedBytes = bos.toByteArray();

                // Lấy nhãn từ folder cha trên HDFS
                String filePath = key.toString();
                String[] parts = filePath.split("/");
                String label = parts[parts.length - 2]; // folder cha là class

                context.write(new Text(label), new BytesWritable(processedBytes));
            }
        }
    }

    // Reducer
    public static class ImageReducer extends Reducer<Text, BytesWritable, Text, BytesWritable> {
        @Override
        protected void reduce(Text key, Iterable<BytesWritable> values, Context context)
                throws IOException, InterruptedException {
            int index = 0;
            for (BytesWritable val : values) {
                String fileName = key.toString() + "_" + index + ".jpg";
                context.write(new Text(fileName), val);
                index++;
            }
        }
    }

    // Hàm đệ quy quét folder HDFS
    public static void addAllHDFSFiles(FileSystem fs, Path path, Job job) throws IOException {
        FileStatus[] statuses = fs.listStatus(path);
        for (FileStatus status : statuses) {
            if (status.isDirectory()) {
                addAllHDFSFiles(fs, status.getPath(), job);
            } else {
                FileInputFormat.addInputPath(job, status.getPath());
            }
        }
    }

    @Override
    public int run(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: ImageNormalizeJob <input_path> <output_path>");
            return -1;
        }

        Configuration conf = getConf();
        Job job = Job.getInstance(conf, "Image Normalize Job");
        job.setJarByClass(ImageNormalizeJob.class);

        // Quét HDFS input recursively
        FileSystem fs = FileSystem.get(conf);
        Path inputPath = new Path(args[0]);
        addAllHDFSFiles(fs, inputPath, job);

        // Mapper + Reducer
        job.setMapperClass(ImageMapper.class);
        job.setReducerClass(ImageReducer.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(BytesWritable.class);

        job.setOutputFormatClass(WholeFileOutputFormat.class);
        WholeFileOutputFormat.setOutputPath(job, new Path(args[1]));

        job.setNumReduceTasks(3);

        return job.waitForCompletion(true) ? 0 : 1;
    }

    public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new ImageNormalizeJob(), args);
        System.exit(exitCode);
    }
}
