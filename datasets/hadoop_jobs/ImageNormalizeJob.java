import java.io.IOException;
import java.io.ByteArrayInputStream;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class ImageNormalizeJob {

    public static class ImageMapper extends Mapper<Text, BytesWritable, Text, Text> {

        @Override
        protected void map(Text key, BytesWritable value, Context context)
                throws IOException, InterruptedException {

            String inputPath = key.toString();
            try {
                // đọc ảnh từ bytes
                ByteArrayInputStream bais = new ByteArrayInputStream(value.getBytes(), 0, value.getLength());
                BufferedImage img = ImageIO.read(bais);

                if (img != null) {
                    // resize về 128x128
                    Image scaled = img.getScaledInstance(128, 128, Image.SCALE_SMOOTH);
                    BufferedImage resized = new BufferedImage(128, 128, BufferedImage.TYPE_INT_RGB);
                    Graphics2D g = resized.createGraphics();
                    g.drawImage(scaled, 0, 0, null);
                    g.dispose();

                    // ghi ảnh ra HDFS
                    String fileName = new Path(inputPath).getName();
                    Path outPath = new Path("/datasets/celeba/normalized_images/" + fileName);
                    FileSystem fs = FileSystem.get(context.getConfiguration());
                    try (FSDataOutputStream out = fs.create(outPath, true)) {
                        ImageIO.write(resized, "jpg", out);
                    }

                    // ghi mapping input → output
                    context.write(new Text(inputPath), new Text(outPath.toString()));
                } else {
                    context.write(new Text(inputPath), new Text("ERROR: Could not decode image"));
                }
            } catch (Exception e) {
                context.write(new Text(inputPath), new Text("ERROR:" + e.getMessage()));
            }
        }
    }

    public static class IdentityReducer extends Reducer<Text, Text, Text, Text> {
        @Override
        protected void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {
            for (Text v : values) {
                context.write(key, v);
            }
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "Image Normalization");

        job.setJarByClass(ImageNormalizeJob.class);
        job.setMapperClass(ImageMapper.class);
        job.setReducerClass(IdentityReducer.class);

        // input format: đọc cả file
        job.setInputFormatClass(WholeFileInputFormat.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        WholeFileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
