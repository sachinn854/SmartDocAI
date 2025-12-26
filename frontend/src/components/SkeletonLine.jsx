export default function SkeletonLine({ width = 'w-full', height = 'h-4' }) {
  return (
    <div className={`${width} ${height} bg-gray-200 dark:bg-gray-700 rounded animate-pulse`}></div>
  );
}
