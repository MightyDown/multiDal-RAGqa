import { createRouter, createWebHashHistory } from 'vue-router'
import KbList from '../views/KbList.vue'
import KbDetail from '../views/KbDetail.vue'
import DocDetail from '../views/DocDetail.vue'
import UploadView from '../views/UploadView.vue'
import QueryView from '../views/QueryView.vue'
import TasksView from '../views/TasksView.vue'

const routes = [
  { path: '/', name: 'kb', component: KbList },
  { path: '/kb/:id', name: 'kb-detail', component: KbDetail, props: true },
  { path: '/kb/:kbId/doc/:taskId', name: 'doc-detail', component: DocDetail, props: true },
  { path: '/upload', name: 'upload', component: UploadView },
  { path: '/query', name: 'query', component: QueryView },
  { path: '/tasks', name: 'tasks', component: TasksView },
]

export default createRouter({
  history: createWebHashHistory(),
  routes,
})
