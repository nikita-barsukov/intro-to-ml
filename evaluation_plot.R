require('ggplot2')
require('ggthemr')
require('reshape')

ggthemr('light')

ds = read.csv('efficiency.csv')
ds = melt(ds)
ds$experiment = seq(1, 8)

pl = ggplot(data = ds,aes(x=experiment, y=value, color=variable)) + 
  geom_line() +
  geom_point() +
  ggtitle('Changing evaluation metrics\nduring developing of classifying algorithm') +
  scale_color_discrete(name='Evaluation\nmetric') +
  scale_y_continuous(name='') +
  scale_x_continuous(name='Iteration')

print(pl)
ggsave('evaluation.png', pl)


